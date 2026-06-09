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
/// A delegating chat client that intercepts agent-as-function tool invocations and surfaces
/// child agent <see cref="ToolApprovalRequestContent"/> items to the parent agent's HITL pipeline.
/// </summary>
internal sealed class AgentAsFunctionApprovalDelegatingChatClient : DelegatingChatClient
{
    /// <summary>
    /// Thread-safe in-memory storage for pending child-agent approvals, keyed by parent session and tool call ID.
    /// </summary>
    private static readonly ConcurrentDictionary<string, AgentAsFunctionPendingApproval> s_pendingApprovals = new(StringComparer.Ordinal);

    internal const string PendingApprovalMarkerPrefix = "__AGENT_AS_FUNCTION_APPROVAL_PENDING__:";
    internal const string ChildCallIdMappingPrefix = "__agent_as_function_child_map__:";

    /// <summary>
    /// Stores a pending child-agent approval in the in-memory dictionary.
    /// </summary>
    internal static string StorePendingApproval(AgentSession parentSession, string parentCallId, AgentAsFunctionPendingApproval pending)
    {
        string key = CreatePendingApprovalKey(parentSession, parentCallId);
        s_pendingApprovals[key] = pending;
        return key;
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

    /// <summary>
    /// Initializes a new instance of the <see cref="AgentAsFunctionApprovalDelegatingChatClient"/> class.
    /// </summary>
    public AgentAsFunctionApprovalDelegatingChatClient(IChatClient innerClient)
        : base(innerClient)
    {
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
            return new ChatResponse(messagesList.Where(static m => m.Role == ChatRole.Assistant && m.Contents.Count > 0).ToList());
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
            foreach (var message in messagesList.Where(static m => m.Role == ChatRole.Assistant && m.Contents.Count > 0))
            {
                yield return new ChatResponseUpdate(message.Role, message.Contents)
                {
                    AuthorName = message.AuthorName,
                    AdditionalProperties = message.AdditionalProperties,
                };
            }

            yield break;
        }

        List<ChatResponseUpdate> allUpdates = [];
        await foreach (var update in base.GetStreamingResponseAsync(messagesList, options, cancellationToken).ConfigureAwait(false))
        {
            allUpdates.Add(update);
        }

        ChatResponse chatResponse = allUpdates.ToChatResponse();
        this.TrySurfacePendingApprovalsInMessages(chatResponse.Messages, session);

        foreach (var message in chatResponse.Messages)
        {
            yield return new ChatResponseUpdate(message.Role, message.Contents)
            {
                AuthorName = message.AuthorName,
                AdditionalProperties = message.AdditionalProperties,
                ResponseId = chatResponse.ResponseId,
                ConversationId = chatResponse.ConversationId,
                CreatedAt = chatResponse.CreatedAt,
                ContinuationToken = chatResponse.ContinuationToken,
                FinishReason = chatResponse.FinishReason,
                RawRepresentation = chatResponse.RawRepresentation,
            };
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

    private bool TrySurfacePendingApprovalsInMessages(IList<ChatMessage> messages, AgentSession session)
    {
        List<(string PendingKey, string ParentCallId, List<ToolApprovalRequestContent> ApprovalRequests)> toSurface = [];

        foreach (var message in messages)
        {
            foreach (var content in message.Contents)
            {
                if (content is FunctionResultContent frc &&
                    frc.Result?.ToString()?.StartsWith(PendingApprovalMarkerPrefix, StringComparison.Ordinal) == true)
                {
                    string pendingKey = frc.Result.ToString()!.Substring(PendingApprovalMarkerPrefix.Length);

                    if (s_pendingApprovals.TryGetValue(pendingKey, out var pending) &&
                        pending?.ApprovalRequests is { Count: > 0 })
                    {
                        string parentCallId = pending.ParentToolCallId ?? GetParentCallIdFromPendingKey(pendingKey);
                        toSurface.Add((pendingKey, parentCallId, pending.ApprovalRequests));
                    }
                }
            }
        }

        if (toSurface.Count == 0)
        {
            return false;
        }

        foreach (var (pendingKey, parentCallId, approvalRequests) in toSurface)
        {
            for (int i = messages.Count - 1; i >= 0; i--)
            {
                var message = messages[i];
                for (int j = message.Contents.Count - 1; j >= 0; j--)
                {
                    if (message.Contents[j] is FunctionResultContent frc &&
                        frc.Result?.ToString() == PendingApprovalMarkerPrefix + pendingKey)
                    {
                        message.Contents.RemoveAt(j);
                    }
                }
            }

            foreach (var message in messages)
            {
                if (message.Role != ChatRole.Assistant)
                {
                    continue;
                }

                for (int k = message.Contents.Count - 1; k >= 0; k--)
                {
                    if (message.Contents[k] is FunctionCallContent fcc && fcc.CallId == parentCallId)
                    {
                        message.Contents.RemoveAt(k);
                        foreach (var approvalRequest in approvalRequests)
                        {
                            message.Contents.Insert(k, approvalRequest);
                            k++;
                        }
                        break;
                    }
                }
            }

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

        for (int i = messages.Count - 1; i >= 0; i--)
        {
            if (messages[i].Contents.Count == 0)
            {
                messages.RemoveAt(i);
            }
        }

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
                    pendingKey is null)
                {
                    continue;
                }

                if (!s_pendingApprovals.TryGetValue(pendingKey, out var pending) || pending is null)
                {
                    continue;
                }

                message.Contents.RemoveAt(j);

                if (approvalResponse.Approved)
                {
                    var resumeMessages = new List<ChatMessage>
                    {
                        new ChatMessage(ChatRole.User, [approvalResponse])
                    };

                    AgentResponse response;
                    try
                    {
                        response = await pending.ChildAgent!.RunAsync(
                            resumeMessages,
                            session: pending.ChildSession,
                            cancellationToken: cancellationToken).ConfigureAwait(false);
                    }
                    catch
                    {
                        s_pendingApprovals.TryRemove(pendingKey, out _);
                        session.StateBag.TryRemoveValue(mappingKey);
                        throw;
                    }

                    var moreApprovals = response.Messages
                        .SelectMany(m => m.Contents)
                        .OfType<ToolApprovalRequestContent>()
                        .ToList();

                    if (moreApprovals.Count > 0)
                    {
                        pending.ApprovalRequests = moreApprovals;
                        pending.ChildMessages = resumeMessages;
                        s_pendingApprovals[pendingKey] = pending;

                        foreach (var req in moreApprovals)
                        {
                            string? childCallId = req.ToolCall?.CallId;
                            if (childCallId is not null)
                            {
                                session.StateBag.SetValue(
                                    ChildCallIdMappingPrefix + childCallId,
                                    pendingKey,
                                    AgentJsonUtilities.DefaultOptions);
                            }
                        }
                    }
                    else
                    {
                        s_pendingApprovals.TryRemove(pendingKey, out _);
                        session.StateBag.TryRemoveValue(mappingKey);

                        messages.Add(new ChatMessage(
                            ChatRole.Tool,
                            [new FunctionResultContent(pending.ParentToolCallId ?? string.Empty, response.Text)]));
                    }
                }
                else
                {
                    s_pendingApprovals.TryRemove(pendingKey, out _);
                    session.StateBag.TryRemoveValue(mappingKey);

                    messages.Add(new ChatMessage(
                        ChatRole.Tool,
                        [new FunctionResultContent(
                            pending.ParentToolCallId ?? string.Empty,
                            "Tool approval was rejected by the user.")]));
                }
            }
        }

        for (int i = messages.Count - 1; i >= 0; i--)
        {
            if (messages[i].Contents.Count == 0)
            {
                messages.RemoveAt(i);
            }
        }
    }

    private static void RemovePendingApprovalRequestsFromMessages(List<ChatMessage> messages, AgentSession session)
    {
        for (int i = messages.Count - 1; i >= 0; i--)
        {
            var message = messages[i];
            for (int j = message.Contents.Count - 1; j >= 0; j--)
            {
                if (message.Contents[j] is not ToolApprovalRequestContent approvalRequest)
                {
                    continue;
                }

                string mappingKey = ChildCallIdMappingPrefix + (approvalRequest.ToolCall?.CallId ?? string.Empty);
                if (session.StateBag.TryGetValue<string>(mappingKey, out var pendingKey, AgentJsonUtilities.DefaultOptions) &&
                    pendingKey is not null)
                {
                    message.Contents.RemoveAt(j);
                }
            }
        }
    }
}
