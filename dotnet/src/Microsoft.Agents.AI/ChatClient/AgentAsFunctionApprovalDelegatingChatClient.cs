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

        // An approval response for a child tool is consumed before the parent model
        // sees the request. If approved, it becomes a parent FunctionResultContent;
        // if rejected, it becomes a rejection result for the same parent call.
        await this.ProcessIncomingApprovalResponsesAsync(messagesList, session, cancellationToken).ConfigureAwait(false);

        // Processing an approval response can resume the child agent only far
        // enough to produce another child approval request. Surface that request
        // immediately instead of asking the parent model to continue.
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

        // Approval responses can synchronously resume a child agent and add another
        // marker result if the child asks for a follow-up approval. Surface that
        // follow-up request without calling the parent model again.
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

        // Streaming uses the same transcript rewrite as non-streaming. Buffering
        // avoids yielding the internal marker FunctionResultContent before we know
        // whether the child agent paused for approval.
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

    /// <summary>
    /// Rewrites pending child-agent approval markers into caller-visible approval requests.
    /// </summary>
    /// <remarks>
    /// The parent <see cref="FunctionInvokingChatClient"/> first records the child agent
    /// as a normal parent tool call followed by an internal marker result. This method
    /// removes only the marker result, keeps the parent tool call for transcript fidelity,
    /// and inserts the child approval request after the parent call.
    /// </remarks>
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
            // The marker is transport-only state. Leaving it in the transcript would
            // make the parent model see an artificial tool result instead of the
            // approval pause.
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

            // Preserve the parent FunctionCallContent so callers can see that the
            // parent agent invoked the child agent tool. The child approval remains
            // the request to answer because its tool call id is what resumes the
            // paused child agent.
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
                        int insertIndex = k + 1;
                        foreach (var approvalRequest in approvalRequests)
                        {
                            message.Contents.Insert(insertIndex, approvalRequest);
                            insertIndex++;
                        }
                        break;
                    }
                }
            }

            // Approval responses arrive with the child tool call id. Map that id
            // back to the stored pending state so approval/rejection can resume or
            // clear the correct child invocation.
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

                // Only approval responses that match a child call id mapping belong
                // to this bridge. Other approval responses are left for the regular
                // approval pipeline.
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
                    // Resume the child with the original child approval response.
                    // Once the child completes, its text becomes the parent tool
                    // result for the original agent-as-function call.
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
                        session.StateBag.TryRemoveValue(mappingKey);

                        // Re-enter the same marker rewrite path used by the initial
                        // child approval. That lets the caller answer the next child
                        // approval before the parent model observes any parent tool result.
                        messages.Add(new ChatMessage(
                            ChatRole.Tool,
                            [new FunctionResultContent(
                                pending.ParentToolCallId ?? string.Empty,
                                PendingApprovalMarkerPrefix + pendingKey)]));
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
