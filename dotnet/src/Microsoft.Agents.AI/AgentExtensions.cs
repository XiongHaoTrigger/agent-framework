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
    private static readonly ConcurrentDictionary<string, WeakReference<PendingAgentToolApprovals>> s_pendingAgentToolApprovalsByRequestId = [];
    private static readonly AsyncLocal<List<ToolApprovalRequestContent>?> s_pendingAgentToolApprovalsForCurrentRun = new();

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

        PendingAgentToolApprovals pendingApprovals = new();

        [Description("Invoke an agent to retrieve some information.")]
        async Task<object?> InvokeAgentAsync(
            [Description("Input query to invoke the agent.")] string query,
            CancellationToken cancellationToken)
        {
            FunctionInvocationContext? parentFunctionContext = FunctionInvokingChatClient.CurrentContext;
            FunctionCallContent? parentToolCall = parentFunctionContext?.CallContent;
            AgentSession? agentSession = session;
            IEnumerable<ChatMessage> inputMessages;

            if (parentToolCall is not null &&
                pendingApprovals.Approvals.TryRemove(parentToolCall.CallId, out PendingAgentToolApproval? pendingApproval))
            {
                _ = s_pendingAgentToolApprovalsByRequestId.TryRemove(parentToolCall.CallId, out _);

                agentSession = pendingApproval.Session;
                inputMessages = pendingApproval.ApprovalRequests.ConvertAll(
                    request => new ChatMessage(ChatRole.User, [request.CreateResponse(approved: true)]));
            }
            else
            {
                inputMessages = [new(ChatRole.User, query)];
            }

            // Propagate any additional properties from the parent agent's run to the child agent if the parent is using a FunctionInvokingChatClient.
            AgentRunOptions? agentRunOptions = FunctionInvokingChatClient.CurrentContext?.Options?.AdditionalProperties is AdditionalPropertiesDictionary dict
                ? new AgentRunOptions { AdditionalProperties = dict }
                : null;

            var response = await agent.RunAsync(inputMessages, session: agentSession, options: agentRunOptions, cancellationToken: cancellationToken).ConfigureAwait(false);

            List<ToolApprovalRequestContent> approvalRequests = response.Messages
                .SelectMany(message => message.Contents)
                .OfType<ToolApprovalRequestContent>()
                .ToList();

            if (approvalRequests.Count > 0 && parentToolCall is not null)
            {
                pendingApprovals.Approvals[parentToolCall.CallId] = new(agent, agentSession, parentToolCall, approvalRequests);
                s_pendingAgentToolApprovalsByRequestId[parentToolCall.CallId] = new(pendingApprovals);

                parentFunctionContext!.Terminate = true;

                s_pendingAgentToolApprovalsForCurrentRun.Value?.Add(new ToolApprovalRequestContent(parentToolCall.CallId, parentToolCall));

                return string.Empty;
            }

            return response.Text;
        }

        options ??= new();
        options.Name ??= SanitizeAgentName(agent.Name);
        options.Description ??= agent.Description;

        return AIFunctionFactory.Create(InvokeAgentAsync, options);
    }

    internal static async Task<IReadOnlyCollection<ChatMessage>> ProcessAgentToolApprovalResponsesAsync(
        IReadOnlyCollection<ChatMessage> messages,
        CancellationToken cancellationToken)
    {
        List<ChatMessage>? processedMessages = null;
        List<AIContent>? toolResults = null;

        foreach (ChatMessage message in messages)
        {
            List<AIContent>? processedContents = null;

            for (int i = 0; i < message.Contents.Count; i++)
            {
                AIContent content = message.Contents[i];
                if (content is ToolApprovalResponseContent { Approved: true } response &&
                    TryTakePendingAgentToolApproval(response.RequestId, out PendingAgentToolApproval? pendingApproval) &&
                    pendingApproval is not null)
                {
                    processedMessages ??= [];
                    processedContents ??= [.. message.Contents.Take(i)];

                    IEnumerable<ChatMessage> approvalMessages = pendingApproval.ApprovalRequests.ConvertAll(
                        request => new ChatMessage(ChatRole.User, [request.CreateResponse(approved: true)]));
                    AgentResponse childResponse = await pendingApproval.Agent.RunAsync(
                        approvalMessages,
                        session: pendingApproval.Session,
                        cancellationToken: cancellationToken).ConfigureAwait(false);

                    (toolResults ??= []).Add(new FunctionResultContent(pendingApproval.ParentToolCall.CallId, childResponse.Text));
                    continue;
                }

                processedContents?.Add(content);
            }

            if (processedMessages is not null)
            {
                if (processedContents is not null)
                {
                    if (processedContents.Count > 0)
                    {
                        ChatMessage processedMessage = message.Clone();
                        processedMessage.Contents = processedContents;
                        processedMessages.Add(processedMessage);
                    }
                }
                else
                {
                    processedMessages.Add(message);
                }
            }
        }

        if (processedMessages is null)
        {
            return messages;
        }

        if (toolResults is { Count: > 0 })
        {
            processedMessages.Add(new ChatMessage(ChatRole.Tool, toolResults));
        }

        return processedMessages;
    }

    internal static IReadOnlyCollection<ChatMessage> RemoveConsumedAgentToolApprovalResponsesFromHistory(
        IReadOnlyCollection<ChatMessage> messages)
    {
        List<ChatMessage>? processedMessages = null;

        foreach (ChatMessage message in messages)
        {
            List<AIContent>? processedContents = null;

            if (message.GetAgentRequestMessageSourceType() == AgentRequestMessageSourceType.ChatHistory)
            {
                for (int i = 0; i < message.Contents.Count; i++)
                {
                    AIContent content = message.Contents[i];
                    if ((content is ToolApprovalResponseContent response &&
                            !HasPendingAgentToolApproval(response.RequestId)) ||
                        (content is ToolApprovalRequestContent request &&
                            !HasPendingAgentToolApproval(request.RequestId)))
                    {
                        processedMessages ??= [];
                        processedContents ??= [.. message.Contents.Take(i)];
                        continue;
                    }

                    processedContents?.Add(content);
                }
            }

            if (processedMessages is not null)
            {
                if (processedContents is not null)
                {
                    if (processedContents.Count > 0)
                    {
                        ChatMessage processedMessage = message.Clone();
                        processedMessage.Contents = processedContents;
                        processedMessages.Add(processedMessage);
                    }
                }
                else
                {
                    processedMessages.Add(message);
                }
            }
        }

        return processedMessages ?? messages;
    }

    internal static bool EnsurePendingAgentToolApprovalsCollectorForCurrentRun()
    {
        if (s_pendingAgentToolApprovalsForCurrentRun.Value is not null)
        {
            return false;
        }

        s_pendingAgentToolApprovalsForCurrentRun.Value = [];
        return true;
    }

    internal static List<ToolApprovalRequestContent>? TakePendingAgentToolApprovalsForCurrentRun()
    {
        List<ToolApprovalRequestContent>? approvalRequests = s_pendingAgentToolApprovalsForCurrentRun.Value;
        s_pendingAgentToolApprovalsForCurrentRun.Value = null;
        return approvalRequests;
    }

    internal static void ClearRejectedAgentToolApprovals(IEnumerable<ChatMessage> messages)
    {
        foreach (ToolApprovalResponseContent response in messages
            .SelectMany(message => message.Contents)
            .OfType<ToolApprovalResponseContent>())
        {
            if (!response.Approved)
            {
                _ = TryTakePendingAgentToolApproval(response.RequestId, out _);
            }
        }
    }

    private static bool TryTakePendingAgentToolApproval(string requestId, out PendingAgentToolApproval? pendingApproval)
    {
        pendingApproval = null;
        if (!s_pendingAgentToolApprovalsByRequestId.TryRemove(requestId, out WeakReference<PendingAgentToolApprovals>? reference) ||
            !reference.TryGetTarget(out PendingAgentToolApprovals? approvals))
        {
            return false;
        }

        return approvals.Approvals.TryRemove(requestId, out pendingApproval);
    }

    private static bool HasPendingAgentToolApproval(string requestId)
    {
        return s_pendingAgentToolApprovalsByRequestId.TryGetValue(requestId, out WeakReference<PendingAgentToolApprovals>? reference) &&
            reference.TryGetTarget(out PendingAgentToolApprovals? approvals) &&
            approvals.Approvals.ContainsKey(requestId);
    }

    private sealed class PendingAgentToolApprovals
    {
        public ConcurrentDictionary<string, PendingAgentToolApproval> Approvals { get; } = [];
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
