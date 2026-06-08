// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.AI;
using Microsoft.Shared.Diagnostics;

namespace Microsoft.Agents.AI;

/// <summary>
/// Provides extensions for <see cref="AIAgent"/>.
/// </summary>
public static partial class AIAgentExtensions
{
    private static readonly ConcurrentDictionary<string, PendingAgentToolApprovals> s_pendingAgentToolApprovalsByRequestId = new(StringComparer.Ordinal);
    private static readonly AsyncLocal<List<ToolApprovalRequestContent>?> s_pendingAgentToolApprovalsForCurrentRun = new();
    private static readonly AsyncLocal<HashSet<string>?> s_consumedAgentToolApprovalRequestIdsForCurrentRun = new();
    private static readonly AsyncLocal<Dictionary<string, Queue<FunctionCallContent>>?> s_streamingAgentToolCallsForCurrentRun = new();
    private static readonly AsyncLocal<HashSet<string>?> s_pendingStreamingAgentToolCallIdsForCurrentRun = new();

    /// <summary>
    /// Creates a new <see cref="AIAgentBuilder"/> using the specified agent as the foundation for the builder pipeline.
    /// </summary>
    /// <param name="innerAgent">The <see cref="AIAgent"/> instance to use as the inner agent.</param>
    /// <returns>A new <see cref="AIAgentBuilder"/> instance configured with the specified inner agent.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="innerAgent"/> is <see langword="null"/>.</exception>
    /// <remarks>
    /// This method provides a convenient way to convert an existing <see cref="AIAgent"/> instance into
    /// a builder pattern, enabling easily wrapping the agent in layers of additional functionality.
    /// It is functionally equivalent to using the <see cref="AIAgentBuilder(AIAgent)"/> constructor directly,
    /// but provides a more fluent API when working with existing agent instances.
    /// </remarks>
    public static AIAgentBuilder AsBuilder(this AIAgent innerAgent)
    {
        _ = Throw.IfNull(innerAgent);

        return new AIAgentBuilder(innerAgent);
    }

    /// <summary>
    /// Creates an <see cref="AIFunction"/> that runs the provided <see cref="AIAgent"/>.
    /// </summary>
    /// <param name="agent">The <see cref="AIAgent"/> to be represented as an invocable function.</param>
    /// <param name="options">
    /// Optional metadata to customize the function representation, such as name and description.
    /// If not provided, defaults will be inferred from the agent's properties.
    /// </param>
    /// <param name="session">
    /// Optional <see cref="AgentSession"/> to use for function invocations. If not provided, a new session
    /// will be created for each function call, which may not preserve conversation context.
    /// </param>
    /// <returns>
    /// An <see cref="AIFunction"/> that can be used as a tool by other agents or AI models to invoke this agent.
    /// </returns>
    /// <exception cref="ArgumentNullException"><paramref name="agent"/> is <see langword="null"/>.</exception>
    /// <remarks>
    /// <para>
    /// This extension method enables agents to participate in function calling scenarios, where they can be
    /// invoked as tools by other agents or AI models. The resulting function accepts a query string as input and
    /// returns the agent's response as a string, making it compatible with standard function calling interfaces
    /// used by AI models.
    /// </para>
    /// <para>
    /// The resulting <see cref="AIFunction"/> is stateful, referencing both the <paramref name="agent"/> and the optional
    /// <paramref name="session"/>. Especially if a specific session is provided, avoid using the resulting function concurrently
    /// in multiple conversations or in requests where the parallel function calls may result in concurrent usage of the session,
    /// as that could lead to undefined and unpredictable behavior.
    /// </para>
    /// </remarks>
    public static AIFunction AsAIFunction(this AIAgent agent, AIFunctionFactoryOptions? options = null, AgentSession? session = null)
    {
        Throw.IfNull(agent);

        [Description("Invoke an agent to retrieve some information.")]
        static string InvokeAgent(
            [Description("Input query to invoke the agent.")] string query,
            CancellationToken cancellationToken) => string.Empty;

        options ??= new();
        options.Name ??= SanitizeAgentName(agent.Name);
        options.Description ??= agent.Description;

        AIFunction declaration = AIFunctionFactory.Create(InvokeAgent, options);
        return new AgentAIFunction(agent, session, declaration);
    }

    private static async ValueTask<object?> InvokeAgentFunctionCoreAsync(
        AIAgent agent,
        AgentSession? session,
        AIFunctionArguments arguments,
        PendingAgentToolApprovals pendingApprovals,
        string agentToolName,
        CancellationToken cancellationToken)
    {
        FunctionInvocationContext? parentFunctionContext = FunctionInvokingChatClient.CurrentContext;
        FunctionCallContent? parentToolCall = parentFunctionContext?.CallContent;
        if (parentToolCall is null &&
            TryTakeStreamingAgentToolCall(agentToolName, out FunctionCallContent? streamingParentToolCall))
        {
            parentToolCall = streamingParentToolCall;
        }

        // Propagate any additional properties from the parent agent's run to the child agent if the parent is using a FunctionInvokingChatClient.
        AgentRunOptions? agentRunOptions = parentFunctionContext?.Options?.AdditionalProperties is AdditionalPropertiesDictionary dict
            ? new AgentRunOptions { AdditionalProperties = dict }
            : null;

        string query = arguments.TryGetValue("query", out object? queryValue) ? queryValue?.ToString() ?? string.Empty : string.Empty;
        AgentResponse response = await agent.RunAsync(query, session: session, options: agentRunOptions, cancellationToken: cancellationToken).ConfigureAwait(false);

        if (parentToolCall is not null &&
            response.Messages.SelectMany(static m => m.Contents).OfType<ToolApprovalRequestContent>().ToList() is { Count: > 0 } approvalRequests)
        {
            AgentSession? childSession = session;
            if (childSession is null && AIAgent.CurrentRunContext?.Agent == agent)
            {
                childSession = AIAgent.CurrentRunContext.Session;
            }

            PendingAgentToolApproval pendingApproval = new(
                agent,
                childSession,
                parentToolCall,
                approvalRequests);

            pendingApprovals.Store(parentToolCall.CallId, pendingApproval);
            s_pendingAgentToolApprovalsByRequestId[parentToolCall.CallId] = pendingApprovals;

            EnsurePendingAgentToolApprovalsCollectorForCurrentRun();
            s_pendingAgentToolApprovalsForCurrentRun.Value!.Add(new ToolApprovalRequestContent(parentToolCall.CallId, parentToolCall));

            if (parentFunctionContext is not null)
            {
                parentFunctionContext.Terminate = true;
            }

            return string.Empty;
        }

        return response.Text;
    }

    /// <summary>
    /// Ensures that the current logical run has a collector for approval requests produced by nested agent-as-tool calls.
    /// </summary>
    /// <returns><see langword="true"/> if this call created the collector; otherwise, <see langword="false"/>.</returns>
    internal static bool EnsurePendingAgentToolApprovalsCollectorForCurrentRun()
    {
        if (s_pendingAgentToolApprovalsForCurrentRun.Value is not null)
        {
            return false;
        }

        s_pendingAgentToolApprovalsForCurrentRun.Value = [];
        s_consumedAgentToolApprovalRequestIdsForCurrentRun.Value = null;
        return true;
    }

    /// <summary>
    /// Gets and clears the approval requests produced by nested agent-as-tool calls in the current logical run.
    /// </summary>
    /// <returns>The approval requests collected for the current logical run.</returns>
    internal static List<ToolApprovalRequestContent> TakePendingAgentToolApprovalsForCurrentRun()
    {
        List<ToolApprovalRequestContent> approvals = s_pendingAgentToolApprovalsForCurrentRun.Value ?? [];
        s_pendingAgentToolApprovalsForCurrentRun.Value = null;
        return approvals;
    }

    /// <summary>
    /// Gets and clears the approval requests produced by nested agent-as-tool calls only when approvals exist.
    /// </summary>
    /// <param name="approvals">The approval requests collected for the current logical run.</param>
    /// <returns><see langword="true"/> if approvals were available; otherwise, <see langword="false"/>.</returns>
    internal static bool TryTakePendingAgentToolApprovalsForCurrentRun(out List<ToolApprovalRequestContent> approvals)
    {
        if (s_pendingAgentToolApprovalsForCurrentRun.Value is { Count: > 0 } pendingApprovals)
        {
            approvals = pendingApprovals;
            s_pendingAgentToolApprovalsForCurrentRun.Value = null;
            return true;
        }

        approvals = [];
        return false;
    }

    /// <summary>
    /// Registers agent-as-tool calls observed while streaming the parent model response.
    /// </summary>
    /// <param name="options">The chat options for the current parent invocation.</param>
    /// <param name="update">The streaming update to inspect.</param>
    /// <returns><see langword="true"/> if the update contains an agent-as-tool call; otherwise, <see langword="false"/>.</returns>
    internal static bool RegisterStreamingAgentToolCalls(ChatOptions? options, ChatResponseUpdate update)
    {
        if (options?.Tools is not { Count: > 0 } tools)
        {
            return false;
        }

        HashSet<string>? agentToolNames = null;
        foreach (AITool tool in tools)
        {
            if (tool is AgentAIFunction agentFunction)
            {
                agentToolNames ??= new(StringComparer.Ordinal);
                agentToolNames.Add(agentFunction.Name);
            }
        }

        if (agentToolNames is null)
        {
            return false;
        }

        bool registeredAny = false;
        foreach (FunctionCallContent functionCall in update.Contents.OfType<FunctionCallContent>())
        {
            if (!agentToolNames.Contains(functionCall.Name))
            {
                continue;
            }

            Dictionary<string, Queue<FunctionCallContent>> callsByName = s_streamingAgentToolCallsForCurrentRun.Value ??= new(StringComparer.Ordinal);
            if (!callsByName.TryGetValue(functionCall.Name, out Queue<FunctionCallContent>? calls))
            {
                calls = new();
                callsByName[functionCall.Name] = calls;
            }

            calls.Enqueue(functionCall);
            s_pendingStreamingAgentToolCallIdsForCurrentRun.Value ??= new(StringComparer.Ordinal);
            s_pendingStreamingAgentToolCallIdsForCurrentRun.Value.Add(functionCall.CallId);
            registeredAny = true;
        }

        return registeredAny;
    }

    /// <summary>
    /// Marks any streamed parent agent-as-tool results in the update as completed.
    /// </summary>
    /// <param name="update">The streaming update to inspect.</param>
    internal static void CompleteStreamingAgentToolCalls(ChatResponseUpdate update)
    {
        HashSet<string>? pendingCallIds = s_pendingStreamingAgentToolCallIdsForCurrentRun.Value;
        if (pendingCallIds is null || pendingCallIds.Count == 0)
        {
            return;
        }

        foreach (FunctionResultContent functionResult in update.Contents.OfType<FunctionResultContent>())
        {
            pendingCallIds.Remove(functionResult.CallId);
        }

        if (pendingCallIds.Count == 0)
        {
            s_pendingStreamingAgentToolCallIdsForCurrentRun.Value = null;
        }
    }

    /// <summary>
    /// Gets whether the current streaming parent invocation still has unfinished agent-as-tool calls.
    /// </summary>
    /// <returns><see langword="true"/> when an observed agent-as-tool call has not yet produced a result.</returns>
    internal static bool HasPendingStreamingAgentToolCallsForCurrentRun()
        => s_pendingStreamingAgentToolCallIdsForCurrentRun.Value is { Count: > 0 };

    /// <summary>
    /// Clears streaming parent agent-as-tool call state for the current logical run.
    /// </summary>
    internal static void ClearStreamingAgentToolCallsForCurrentRun()
    {
        s_streamingAgentToolCallsForCurrentRun.Value = null;
        s_pendingStreamingAgentToolCallIdsForCurrentRun.Value = null;
    }

    /// <summary>
    /// Processes approval responses that target pending nested agent-as-tool calls and replaces approved responses
    /// with the parent tool-call result messages needed to continue the outer agent invocation.
    /// </summary>
    /// <param name="inputMessages">The messages supplied to the outer agent run.</param>
    /// <param name="cancellationToken">The token used to cancel the operation.</param>
    /// <returns>The input messages with consumed agent-as-tool approval responses removed and resumed tool results appended.</returns>
    internal static async Task<IReadOnlyCollection<ChatMessage>> ProcessAgentToolApprovalResponsesAsync(
        IReadOnlyCollection<ChatMessage> inputMessages,
        CancellationToken cancellationToken)
    {
        List<ChatMessage>? processedMessages = null;
        List<ChatMessage>? resumedToolResultMessages = null;

        foreach (ChatMessage message in inputMessages)
        {
            List<AIContent>? processedContents = null;

            for (int i = 0; i < message.Contents.Count; i++)
            {
                AIContent content = message.Contents[i];
                if (content is ToolApprovalResponseContent approvalResponse &&
                    TryTakePendingAgentToolApproval(approvalResponse.RequestId, out PendingAgentToolApproval? pendingApproval))
                {
                    processedContents ??= [.. message.Contents.Take(i)];
                    if (pendingApproval is null)
                    {
                        throw new InvalidOperationException("A pending agent approval was expected.");
                    }

                    AddConsumedAgentToolApprovalRequestId(approvalResponse.RequestId);

                    if (approvalResponse.Approved)
                    {
                        // Resume the child agent with its own approval response, then surface the completed child
                        // response as the result of the parent tool call.
                        AgentResponse childResponse = await pendingApproval.Agent.RunAsync(
                            pendingApproval.ApprovalRequests.Select(r => new ChatMessage(ChatRole.User, [r.CreateResponse(approved: true)])),
                            pendingApproval.Session,
                            cancellationToken: cancellationToken).ConfigureAwait(false);

                        resumedToolResultMessages ??= [];
                        resumedToolResultMessages.Add(new ChatMessage(ChatRole.Assistant, [pendingApproval.ParentToolCall]));
                        resumedToolResultMessages.Add(new ChatMessage(ChatRole.Tool, [new FunctionResultContent(pendingApproval.ParentToolCall.CallId, childResponse.Text)]));
                    }
                    else
                    {
                        // A rejected child approval completes the parent tool call with a rejection result,
                        // allowing the parent model to continue without executing the protected child tool.
                        string rejectionResult = string.IsNullOrWhiteSpace(approvalResponse.Reason)
                            ? "The tool call was rejected by the caller."
                            : $"The tool call was rejected by the caller: {approvalResponse.Reason}";

                        resumedToolResultMessages ??= [];
                        resumedToolResultMessages.Add(new ChatMessage(ChatRole.Assistant, [pendingApproval.ParentToolCall]));
                        resumedToolResultMessages.Add(new ChatMessage(ChatRole.Tool, [new FunctionResultContent(pendingApproval.ParentToolCall.CallId, rejectionResult)]));
                    }
                }
                else
                {
                    processedContents?.Add(content);
                }
            }

            if (processedContents is not null)
            {
                processedMessages ??= [];
                if (processedContents.Count > 0)
                {
                    ChatMessage processedMessage = message.Clone();
                    processedMessage.Contents = processedContents;
                    processedMessages.Add(processedMessage);
                }
            }
            else
            {
                processedMessages?.Add(message);
            }
        }

        if (resumedToolResultMessages is not null)
        {
            processedMessages ??= [.. inputMessages];
            processedMessages.AddRange(resumedToolResultMessages);
        }

        return processedMessages ?? inputMessages;
    }

    /// <summary>
    /// Removes consumed agent-as-tool approval messages from chat history before the outer chat client sees them.
    /// </summary>
    /// <param name="messages">The prepared message list for the outer chat client invocation.</param>
    internal static void RemoveConsumedAgentToolApprovalResponsesFromHistory(List<ChatMessage> messages)
    {
        HashSet<string>? consumedApprovalRequestIds = s_consumedAgentToolApprovalRequestIdsForCurrentRun.Value;
        if (consumedApprovalRequestIds is null || consumedApprovalRequestIds.Count == 0)
        {
            return;
        }

        for (int i = messages.Count - 1; i >= 0; i--)
        {
            ChatMessage message = messages[i];
            if (message.GetAgentRequestMessageSourceType() != AgentRequestMessageSourceType.ChatHistory)
            {
                continue;
            }

            bool removedAny = false;
            List<AIContent> contents = [.. message.Contents];
            for (int j = contents.Count - 1; j >= 0; j--)
            {
                if (contents[j] is ToolApprovalRequestContent request &&
                    consumedApprovalRequestIds.Contains(request.RequestId))
                {
                    contents.RemoveAt(j);
                    removedAny = true;
                }
                else if (contents[j] is ToolApprovalResponseContent response &&
                    consumedApprovalRequestIds.Contains(response.RequestId))
                {
                    contents.RemoveAt(j);
                    removedAny = true;
                }
            }

            if (!removedAny)
            {
                continue;
            }

            if (contents.Count == 0)
            {
                messages.RemoveAt(i);
            }
            else
            {
                ChatMessage clonedMessage = message.Clone();
                clonedMessage.Contents = contents;
                messages[i] = clonedMessage;
            }
        }

        s_consumedAgentToolApprovalRequestIdsForCurrentRun.Value = null;
    }

    private static bool TryTakePendingAgentToolApproval(string requestId, out PendingAgentToolApproval? approval)
    {
        approval = null;
        if (!s_pendingAgentToolApprovalsByRequestId.TryGetValue(requestId, out PendingAgentToolApprovals? approvals))
        {
            return false;
        }

        if (!approvals.TryTake(requestId, out approval))
        {
            return false;
        }

        s_pendingAgentToolApprovalsByRequestId.TryRemove(requestId, out _);
        return true;
    }

    private static void AddConsumedAgentToolApprovalRequestId(string requestId)
    {
        s_consumedAgentToolApprovalRequestIdsForCurrentRun.Value ??= new(StringComparer.Ordinal);
        s_consumedAgentToolApprovalRequestIdsForCurrentRun.Value.Add(requestId);
    }

    private static bool TryTakeStreamingAgentToolCall(string agentToolName, out FunctionCallContent? parentToolCall)
    {
        parentToolCall = null;
        Dictionary<string, Queue<FunctionCallContent>>? callsByName = s_streamingAgentToolCallsForCurrentRun.Value;
        if (callsByName is null ||
            !callsByName.TryGetValue(agentToolName, out Queue<FunctionCallContent>? calls) ||
            calls.Count == 0)
        {
            return false;
        }

        parentToolCall = calls.Dequeue();
        if (calls.Count == 0)
        {
            callsByName.Remove(agentToolName);
        }

        if (callsByName.Count == 0)
        {
            s_streamingAgentToolCallsForCurrentRun.Value = null;
        }

        return true;
    }

    private sealed class AgentAIFunction(AIAgent agent, AgentSession? session, AIFunction innerFunction)
        : DelegatingAIFunction(innerFunction)
    {
        private readonly PendingAgentToolApprovals _pendingApprovals = new();

        protected override ValueTask<object?> InvokeCoreAsync(AIFunctionArguments arguments, CancellationToken cancellationToken)
            => InvokeAgentFunctionCoreAsync(agent, session, arguments, this._pendingApprovals, this.Name, cancellationToken);
    }

    private sealed class PendingAgentToolApprovals
    {
        private readonly object _gate = new();
        private readonly Dictionary<string, PendingAgentToolApproval> _approvals = new(StringComparer.Ordinal);

        public void Store(string requestId, PendingAgentToolApproval approval)
        {
            lock (this._gate)
            {
                this._approvals[requestId] = approval;
            }
        }

        public bool TryTake(string requestId, out PendingAgentToolApproval? approval)
        {
            lock (this._gate)
            {
                if (!this._approvals.TryGetValue(requestId, out PendingAgentToolApproval? pendingApproval))
                {
                    approval = null;
                    return false;
                }

                this._approvals.Remove(requestId);
                approval = pendingApproval;
                return true;
            }
        }

        public void Remove(string requestId)
        {
            lock (this._gate)
            {
                this._approvals.Remove(requestId);
            }
        }
    }

    private sealed record PendingAgentToolApproval(
        AIAgent Agent,
        AgentSession? Session,
        FunctionCallContent ParentToolCall,
        List<ToolApprovalRequestContent> ApprovalRequests);

    /// <summary>
    /// Removes characters from AI agent name that shouldn't be used in an AI function name.
    /// </summary>
    /// <param name="agentName">The AI agent name to sanitize.</param>
    /// <returns>
    /// The sanitized agent name with invalid characters replaced by underscores, or <c>null</c> if the input is <c>null</c>.
    /// </returns>
    private static string? SanitizeAgentName(string? agentName)
    {
        return agentName is null
            ? agentName
            : InvalidNameCharsRegex().Replace(agentName, "_");
    }

    /// <summary>Regex that flags any character other than ASCII digits or letters.</summary>
#if NET
    [GeneratedRegex("[^0-9A-Za-z]+")]
    private static partial Regex InvalidNameCharsRegex();
#else
    private static Regex InvalidNameCharsRegex() => s_invalidNameCharsRegex;
    private static readonly Regex s_invalidNameCharsRegex = new("[^0-9A-Za-z]+", RegexOptions.Compiled);
#endif
}
