// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.AI;

namespace Microsoft.Agents.AI.ChatClient;

internal sealed class AgentAsFunctionApprovalForwardingChatClient : DelegatingChatClient
{
    private const string s_continuationPrefix = "__agent_continuation__:";

    public AgentAsFunctionApprovalForwardingChatClient(IChatClient innerClient) : base(innerClient) { }

    public override async Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
        CancellationToken cancellationToken = new CancellationToken())
    {
        // 唤醒子agent执行

        var response = await base.GetResponseAsync(messages, options, cancellationToken).ConfigureAwait(false);

        // 检查返回值
        var session = AIAgent.CurrentRunContext?.Session;
        session = session ?? throw new InvalidOperationException("Session is null");

        var isSucceed = session.StateBag.TryGetValue<Dictionary<string, AgentAsFunctionContinuation>>(
            AgentAsFunctionContinuation.StateBagKey,
            out var continuations,
            AgentJsonUtilities.DefaultOptions);

        if (isSucceed == false || continuations is null)
        {
            return response;
        }

        foreach (var message in response.Messages)
        {
            List<AIContent> replaceContents = [];
            for (int i = 0; i < message.Contents.Count; i++)
            {
                var content = message.Contents[i];
                if (TryGetContinuationId(content, out var continuationId) &&
                    continuations.TryGetValue(continuationId, out var continuation))
                {
                    replaceContents.AddRange(continuation.PendingToolApprovalRequestDict.Values);
                }
                else
                {
                    replaceContents.Add(content);
                }
            }
            message.Contents = replaceContents;
        }
        return response;
    }

    private static bool TryGetContinuationId(
        AIContent content,
        out string continuationId)
    {
        continuationId = string.Empty;

        if (content is not FunctionResultContent { Result: string result } ||
            !result.StartsWith("__agent_continuation__:", StringComparison.Ordinal))
        {
            return false;
        }

        continuationId = result[s_continuationPrefix.Length..];
        return continuationId.Length > 0;
    }
}
