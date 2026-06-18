// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.AI;

namespace Microsoft.Agents.AI;

/// <summary>
/// Surfaces approval requests from child agents invoked as function tools to the parent agent.
/// </summary>
internal sealed class AgentAsFunctionApprovalDelegatingChatClient : DelegatingChatClient
{
    private static readonly ConcurrentDictionary<string, AgentAsFunctionPendingApproval> s_pendingApprovals = new(StringComparer.Ordinal);

    internal const string PendingApprovalMarkerPrefix = "__AGENT_AS_FUNCTION_APPROVAL_PENDING__:";
    internal const string ChildCallIdMappingPrefix = "__agent_as_function_child_map__:";

    /// <summary>
    /// Initializes a new instance of the <see cref="AgentAsFunctionApprovalDelegatingChatClient"/> class.
    /// </summary>
    public AgentAsFunctionApprovalDelegatingChatClient(IChatClient innerClient)
        : base(innerClient)
    {
    }

    /// <summary>
    /// Stores a pending child-agent approval request.
    /// </summary>
    internal static string StorePendingApproval(
        AgentSession parentSession,
        string parentCallId,
        AgentAsFunctionPendingApproval pending)
    {
        string key = CreatePendingApprovalKey(parentSession, parentCallId);
        s_pendingApprovals[key] = pending;
        return key;
    }

    /// <inheritdoc/>
    public override async Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var session = GetRequiredSession();
        var messagesList = messages.ToList();

        await this.ProcessIncomingApprovalResponsesAsync(messagesList, session, cancellationToken).ConfigureAwait(false);

        if (this.TrySurfacePendingApprovalsInMessages(messagesList, session))
        {
            return new ChatResponse(GetAssistantMessagesWithContent(messagesList).ToList());
        }

        var response = await base.GetResponseAsync(messagesList, options, cancellationToken).ConfigureAwait(false);
        this.TrySurfacePendingApprovalsInMessages(response.Messages, session);
        return response;
    }

    /// <inheritdoc/>
    public override async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var session = GetRequiredSession();
        var messagesList = messages.ToList();

        await this.ProcessIncomingApprovalResponsesAsync(messagesList, session, cancellationToken).ConfigureAwait(false);

        if (this.TrySurfacePendingApprovalsInMessages(messagesList, session))
        {
            foreach (var message in GetAssistantMessagesWithContent(messagesList))
            {
                yield return CreateUpdate(message);
            }

            yield break;
        }

        List<ChatResponseUpdate> updates = [];
        await foreach (var update in base.GetStreamingResponseAsync(messagesList, options, cancellationToken).ConfigureAwait(false))
        {
            updates.Add(update);
        }

        ChatResponse response = updates.ToChatResponse();
        this.TrySurfacePendingApprovalsInMessages(response.Messages, session);

        foreach (var message in response.Messages)
        {
            yield return CreateUpdate(message, response);
        }
    }

    private static AgentSession GetRequiredSession()
    {
        var runContext = AIAgent.CurrentRunContext
            ?? throw new InvalidOperationException(
                $"{nameof(AgentAsFunctionApprovalDelegatingChatClient)} can only be used within the context of a running AIAgent.");

        return runContext.Session
            ?? throw new InvalidOperationException(
                $"{nameof(AgentAsFunctionApprovalDelegatingChatClient)} requires a session.");
    }

    private static string CreatePendingApprovalKey(AgentSession parentSession, string parentCallId)
        => $"{RuntimeHelpers.GetHashCode(parentSession):X8}:{parentCallId}";

    private static string GetParentCallIdFromPendingKey(string pendingKey)
    {
        int separatorIndex = pendingKey.IndexOf(':');
        return separatorIndex >= 0 && separatorIndex + 1 < pendingKey.Length
            ? pendingKey.Substring(separatorIndex + 1)
            : pendingKey;
    }

    private static IEnumerable<ChatMessage> GetAssistantMessagesWithContent(IEnumerable<ChatMessage> messages)
        => messages.Where(static message => message.Role == ChatRole.Assistant && message.Contents.Count > 0);

    private static ChatResponseUpdate CreateUpdate(ChatMessage message, ChatResponse? response = null)
        => new(message.Role, message.Contents)
        {
            AuthorName = message.AuthorName,
            AdditionalProperties = message.AdditionalProperties,
            ResponseId = response?.ResponseId,
            ConversationId = response?.ConversationId,
            CreatedAt = response?.CreatedAt,
            ContinuationToken = response?.ContinuationToken,
            FinishReason = response?.FinishReason,
            RawRepresentation = response?.RawRepresentation,
        };

    private bool TrySurfacePendingApprovalsInMessages(IList<ChatMessage> messages, AgentSession session)
    {
        List<(string PendingKey, string ParentCallId, List<ToolApprovalRequestContent> ApprovalRequests)> approvalsToSurface = [];

        foreach (var message in messages)
        {
            foreach (var content in message.Contents)
            {
                if (content is FunctionResultContent functionResult &&
                    functionResult.Result?.ToString()?.StartsWith(PendingApprovalMarkerPrefix, StringComparison.Ordinal) == true)
                {
                    string pendingKey = functionResult.Result.ToString()![PendingApprovalMarkerPrefix.Length..];

                    if (s_pendingApprovals.TryGetValue(pendingKey, out var pending) &&
                        pending.ApprovalRequests.Count > 0)
                    {
                        string parentCallId = pending.ParentToolCallId.Length > 0
                            ? pending.ParentToolCallId
                            : GetParentCallIdFromPendingKey(pendingKey);

                        approvalsToSurface.Add((pendingKey, parentCallId, pending.ApprovalRequests));
                    }
                }
            }
        }

        if (approvalsToSurface.Count == 0)
        {
            return false;
        }

        foreach (var (pendingKey, parentCallId, approvalRequests) in approvalsToSurface)
        {
            RemovePendingApprovalMarker(messages, pendingKey);
            InsertApprovalsAfterParentCall(messages, parentCallId, approvalRequests);
            StoreChildCallMappings(session, pendingKey, approvalRequests);
        }

        RemoveEmptyMessages(messages);
        return true;
    }

    private async Task ProcessIncomingApprovalResponsesAsync(
        List<ChatMessage> messages,
        AgentSession session,
        CancellationToken cancellationToken)
    {
        RemovePendingApprovalRequestsFromMessages(messages, session);

        for (int i = messages.Count - 1; i >= 0; i--)
        {
            var message = messages[i];
            for (int j = message.Contents.Count - 1; j >= 0; j--)
            {
                if (message.Contents[j] is not ToolApprovalResponseContent approvalResponse)
                {
                    continue;
                }

                string mappingKey = ChildCallIdMappingPrefix + (approvalResponse.ToolCall?.CallId ?? string.Empty);
                if (!session.StateBag.TryGetValue<string>(mappingKey, out var pendingKey, AgentJsonUtilities.DefaultOptions) ||
                    pendingKey is null ||
                    !s_pendingApprovals.TryGetValue(pendingKey, out var pending))
                {
                    continue;
                }

                message.Contents.RemoveAt(j);

                if (approvalResponse.Approved)
                {
                    await this.ResumeApprovedChildAgentAsync(
                        messages,
                        session,
                        mappingKey,
                        pendingKey,
                        pending,
                        approvalResponse,
                        cancellationToken).ConfigureAwait(false);
                }
                else
                {
                    CompletePendingApproval(session, mappingKey, pendingKey);
                    AddParentToolResult(messages, pending.ParentToolCallId, "Tool approval was rejected by the user.");
                }
            }
        }

        RemoveEmptyMessages(messages);
    }

    private async Task ResumeApprovedChildAgentAsync(
        List<ChatMessage> messages,
        AgentSession session,
        string mappingKey,
        string pendingKey,
        AgentAsFunctionPendingApproval pending,
        ToolApprovalResponseContent approvalResponse,
        CancellationToken cancellationToken)
    {
        var resumeMessages = new List<ChatMessage>
        {
            new(ChatRole.User, [approvalResponse])
        };

        AgentResponse response;
        try
        {
            response = await pending.ChildAgent.RunAsync(
                resumeMessages,
                session: pending.ChildSession,
                cancellationToken: cancellationToken).ConfigureAwait(false);
        }
        catch
        {
            CompletePendingApproval(session, mappingKey, pendingKey);
            throw;
        }

        var approvalRequests = response.Messages
            .SelectMany(static m => m.Contents)
            .OfType<ToolApprovalRequestContent>()
            .ToList();

        if (approvalRequests.Count > 0)
        {
            pending.ApprovalRequests = approvalRequests;
            s_pendingApprovals[pendingKey] = pending;
            session.StateBag.TryRemoveValue(mappingKey);
            AddPendingApprovalMarker(messages, pending.ParentToolCallId, pendingKey);
        }
        else
        {
            CompletePendingApproval(session, mappingKey, pendingKey);
            AddParentToolResult(messages, pending.ParentToolCallId, response.Text);
        }
    }

    private static void CompletePendingApproval(AgentSession session, string mappingKey, string pendingKey)
    {
        s_pendingApprovals.TryRemove(pendingKey, out _);
        session.StateBag.TryRemoveValue(mappingKey);
    }

    private static void RemovePendingApprovalMarker(IList<ChatMessage> messages, string pendingKey)
    {
        string marker = PendingApprovalMarkerPrefix + pendingKey;

        foreach (var message in messages)
        {
            for (int i = message.Contents.Count - 1; i >= 0; i--)
            {
                if (message.Contents[i] is FunctionResultContent functionResult &&
                    functionResult.Result?.ToString() == marker)
                {
                    message.Contents.RemoveAt(i);
                }
            }
        }
    }

    private static void InsertApprovalsAfterParentCall(
        IList<ChatMessage> messages,
        string parentCallId,
        List<ToolApprovalRequestContent> approvalRequests)
    {
        foreach (var message in messages)
        {
            if (message.Role != ChatRole.Assistant)
            {
                continue;
            }

            for (int i = message.Contents.Count - 1; i >= 0; i--)
            {
                if (message.Contents[i] is FunctionCallContent functionCall && functionCall.CallId == parentCallId)
                {
                    int insertIndex = i + 1;
                    foreach (var approvalRequest in approvalRequests)
                    {
                        message.Contents.Insert(insertIndex++, approvalRequest);
                    }

                    return;
                }
            }
        }
    }

    private static void StoreChildCallMappings(
        AgentSession session,
        string pendingKey,
        List<ToolApprovalRequestContent> approvalRequests)
    {
        foreach (var approvalRequest in approvalRequests)
        {
            string? childCallId = approvalRequest.ToolCall?.CallId;
            if (childCallId is not null)
            {
                session.StateBag.SetValue(
                    ChildCallIdMappingPrefix + childCallId,
                    pendingKey,
                    AgentJsonUtilities.DefaultOptions);
            }
        }
    }

    private static void RemovePendingApprovalRequestsFromMessages(List<ChatMessage> messages, AgentSession session)
    {
        foreach (var message in messages)
        {
            for (int i = message.Contents.Count - 1; i >= 0; i--)
            {
                if (message.Contents[i] is not ToolApprovalRequestContent approvalRequest)
                {
                    continue;
                }

                string mappingKey = ChildCallIdMappingPrefix + (approvalRequest.ToolCall?.CallId ?? string.Empty);
                if (session.StateBag.TryGetValue<string>(mappingKey, out var pendingKey, AgentJsonUtilities.DefaultOptions) &&
                    pendingKey is not null)
                {
                    message.Contents.RemoveAt(i);
                }
            }
        }
    }

    private static void AddPendingApprovalMarker(List<ChatMessage> messages, string parentCallId, string pendingKey)
        => AddParentToolResult(messages, parentCallId, PendingApprovalMarkerPrefix + pendingKey);

    private static void AddParentToolResult(List<ChatMessage> messages, string parentCallId, object? result)
        => messages.Add(new ChatMessage(
            ChatRole.Tool,
            [new FunctionResultContent(parentCallId, result)]));

    private static void RemoveEmptyMessages(IList<ChatMessage> messages)
    {
        for (int i = messages.Count - 1; i >= 0; i--)
        {
            if (messages[i].Contents.Count == 0)
            {
                messages.RemoveAt(i);
            }
        }
    }
}
