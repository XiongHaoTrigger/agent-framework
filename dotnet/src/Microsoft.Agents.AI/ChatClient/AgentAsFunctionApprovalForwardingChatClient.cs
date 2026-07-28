// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.AI;

namespace Microsoft.Agents.AI.ChatClient;

internal sealed class AgentAsFunctionApprovalForwardingChatClient : DelegatingChatClient
{
    private const string ContinuationPrefix = "__agent_continuation__:";

    public AgentAsFunctionApprovalForwardingChatClient(IChatClient innerClient) : base(innerClient) { }

    public override async Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        // 必须在调用内部客户端前捕获父 Session；内部调用可能切换 CurrentRunContext。
        var parentSession = AIAgent.CurrentRunContext?.Session;

        // 如果消息中包含当前待处理的审批响应，先恢复对应的子 Agent。
        var messageList = messages.ToList();
        if (parentSession is not null)
        {
            var resumeResult = await this.TryResumeChildAgentAsync(messageList, options, parentSession, cancellationToken)
                .ConfigureAwait(false);
            if (resumeResult?.ApprovalRequest is not null)
            {
                // 子 Agent 再次请求审批，不能继续父 Agent。
                return resumeResult.ApprovalRequest;
            }

            if (resumeResult?.ParentMessages is not null)
            {
                // 子 Agent 已完成，继续父 Agent。
                messageList = resumeResult.ParentMessages;
            }
        }

        var response = await base.GetResponseAsync(messageList, options, cancellationToken).ConfigureAwait(false);

        if (parentSession is null)
        {
            return response;
        }

        var isSucceed = parentSession.StateBag.TryGetValue<Dictionary<string, AgentAsFunctionContinuation>>(
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

    public override async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        // 必须在开始调用内部客户端前保存父 Session。
        var parentSession = AIAgent.CurrentRunContext?.Session;
        var messageList = messages.ToList();

        if (parentSession is not null)
        {
            // streaming 和非 streaming 共用同一套子 Agent 恢复逻辑。
            var resumeResult = await this.TryResumeChildAgentAsync(messageList, options, parentSession, cancellationToken).ConfigureAwait(false);

            if (resumeResult?.ApprovalRequest is not null)
            {
                // 子 Agent 再次请求审批，直接把审批响应转换成 streaming update。
                foreach (var update in new AgentResponse(resumeResult.ApprovalRequest).ToAgentResponseUpdates())
                {
                    yield return update.AsChatResponseUpdate();
                }

                yield break;
            }

            if (resumeResult?.ParentMessages is not null)
            {
                // 子 Agent 已完成，把 FunctionResultContent 交给父 Agent 继续运行。
                messageList = resumeResult.ParentMessages;
            }
        }

        await foreach (var update in base.GetStreamingResponseAsync(messageList, options, cancellationToken).ConfigureAwait(false))
        {
            if (parentSession is not null && TryReplaceContinuationMarkers(update.Contents, parentSession) is { } rewrittenContents)
            {
                update.Contents = rewrittenContents;
            }

            yield return update;
        }
    }

    /// <summary>
    /// 检查当前的content是否是marker，并且获取对应的continuationId即parent agent 对 sub agent 发起调用的function call id
    /// </summary>
    /// <param name="content">AIContent</param>
    /// <param name="continuationId">parent agent 对 sub agent 发起调用的function call id</param>
    /// <returns>是否获取成功</returns>
    private static bool TryGetContinuationId(AIContent content, out string continuationId)
    {
        continuationId = string.Empty;

        if (content is not FunctionResultContent { Result: { } rawResult })
        {
            return false;
        }

        var result = rawResult switch
        {
            string text => text,
            JsonElement { ValueKind: JsonValueKind.String } json =>
                json.GetString(),
            _ => null,
        };

        if (result?.StartsWith(
                ContinuationPrefix,
                StringComparison.Ordinal) is not true)
        {
            return false;
        }
        continuationId = result[ContinuationPrefix.Length..];
        return continuationId.Length > 0;
    }

    private async Task<ChildResumeResult?> TryResumeChildAgentAsync(List<ChatMessage> messages, ChatOptions? options,
        AgentSession parentSession, CancellationToken cancellationToken)
    {
        // continuation 保存了正在等待的审批请求以及对应的子 Agent Session。
        if (!parentSession.StateBag.TryGetValue<Dictionary<string, AgentAsFunctionContinuation>>(
                AgentAsFunctionContinuation.StateBagKey,
                out var continuations,
                AgentJsonUtilities.DefaultOptions) || continuations is null)
        {
            return null;
        }

        // 收集调用者提交的所有审批响应。
        var approvalResponses = messages
            .SelectMany(message => message.Contents)
            .OfType<ToolApprovalResponseContent>()
            .ToList();

        if (approvalResponses.Count == 0)
        {
            return null;
        }

        var continuationEntry = continuations.FirstOrDefault(pair =>
            approvalResponses.Any(response =>
                pair.Value.PendingToolApprovalRequestDict.ContainsKey(response.RequestId)));

        if (continuationEntry.Value is null)
        {
            return null;
        }

        var continuationId = continuationEntry.Key;
        var continuation = continuationEntry.Value;

        // 同一个子 Agent 可能在一轮中同时产生多个审批请求。
        // 收集属于当前 continuation 的全部审批响应，并忽略重复的 RequestId。
        var reboundApprovals = new List<ToolApprovalResponseContent>();
        var processedRequestIds = new HashSet<string>(StringComparer.Ordinal);

        foreach (var approvalResponse in approvalResponses)
        {
            if (!processedRequestIds.Add(approvalResponse.RequestId) ||
                !continuation.PendingToolApprovalRequestDict.TryGetValue(
                    approvalResponse.RequestId,
                    out var originalRequest))
            {
                continue;
            }

            // 使用原始审批请求中的 ToolCall，不能信任用户传入的 ToolCall。
            reboundApprovals.Add(new ToolApprovalResponseContent(
                approvalResponse.RequestId,
                approvalResponse.Approved,
                originalRequest.ToolCall)
            {
                Reason = approvalResponse.Reason,
            });
        }

        // StateBag 必须可序列化，不能直接保存 Agent 引用。
        // 因此使用父函数调用名，从本次可用工具中找回原来的 AgentAIFunction。
        var ficc = this.GetService<FunctionInvokingChatClient>();
        var agentFunction = (options?.Tools ?? Enumerable.Empty<AITool>())
            .Concat(ficc?.AdditionalTools ?? Enumerable.Empty<AITool>())
            .OfType<AIFunction>()
            .Select(tool => tool.GetService<AgentAIFunction>())
            .FirstOrDefault(tool =>
                tool is not null && string.Equals(tool.Name, continuation.ParentCallName, StringComparison.Ordinal));

        if (agentFunction is null)
        {
            throw new InvalidOperationException(
                $"Cannot find agent function '{continuation.ParentCallName}'.");
        }

        // 使用保存的状态恢复原子 Agent 的 Session，而不是创建一个新 Session。
        var childSession = await agentFunction.Agent.DeserializeSessionAsync(
            continuation.SerializedSession,
            AgentJsonUtilities.DefaultOptions,
            cancellationToken).ConfigureAwait(false);

        // 把本轮全部审批结果作为一条用户消息传回子 Agent，使它从暂停位置继续执行。
        var childResponse = await RunChildAgentAsync(agentFunction.Agent, childSession, reboundApprovals, cancellationToken)
            .ConfigureAwait(false);

        var newApprovalRequests = childResponse.Messages
            .SelectMany(message => message.Contents)
            .OfType<ToolApprovalRequestContent>()
            .ToList();

        if (newApprovalRequests.Count > 0)
        {
            // 子 Agent 再次请求审批：保存最新 Session，并用新请求替换当前 pending 请求。
            var serializedSession = await agentFunction.Agent
                .SerializeSessionAsync(childSession, AgentJsonUtilities.DefaultOptions, cancellationToken)
                .ConfigureAwait(false);
            var surfacedRequestIds = new HashSet<string>(
                continuation.SurfacedToolApprovalRequestIds,
                StringComparer.Ordinal);
            surfacedRequestIds.UnionWith(newApprovalRequests.Select(request => request.RequestId));

            continuations[continuationId] = continuation with
            {
                SerializedSession = serializedSession,
                PendingToolApprovalRequestDict = new ConcurrentDictionary<string, ToolApprovalRequestContent>(
                    newApprovalRequests.ToDictionary(request => request.RequestId, request => request)),
                SurfacedToolApprovalRequestIds = surfacedRequestIds,
            };

            parentSession.StateBag.SetValue(AgentAsFunctionContinuation.StateBagKey, continuations, AgentJsonUtilities.DefaultOptions);

            var approvalMessages = childResponse.Messages
                .Select(message =>
                {
                    var clone = message.Clone();
                    clone.Contents = message.Contents
                        .OfType<ToolApprovalRequestContent>()
                        .Cast<AIContent>()
                        .ToList();

                    return clone;
                })
                .Where(message => message.Contents.Count > 0)
                .ToList();

            return new ChildResumeResult(ApprovalRequest: new ChatResponse(approvalMessages));
        }

        // 子 Agent 已完成，不再需要保存 continuation。
        continuations.Remove(continuationId);

        if (continuations.Count == 0)
        {
            parentSession.StateBag.TryRemoveValue(AgentAsFunctionContinuation.StateBagKey);
        }
        else
        {
            parentSession.StateBag.SetValue(AgentAsFunctionContinuation.StateBagKey, continuations, AgentJsonUtilities.DefaultOptions);
        }

        // 消息历史会包含前几轮审批。最终回到父 Agent 前，要一次性删除属于该子 Agent 的全部审批消息。
        // 同时合并当前 pending ID，以兼容没有累计字段的旧 continuation。
        var surfacedRequestIdsToRemove = new HashSet<string>(
            continuation.SurfacedToolApprovalRequestIds,
            StringComparer.Ordinal);
        surfacedRequestIdsToRemove.UnionWith(continuation.PendingToolApprovalRequestDict.Keys);
        var rewrittenMessages = new List<ChatMessage>();

        foreach (var message in messages)
        {
            var contents = message.Contents
                .Where(content => content switch
                {
                    ToolApprovalRequestContent request =>
                        !surfacedRequestIdsToRemove.Contains(request.RequestId),
                    ToolApprovalResponseContent response =>
                        !surfacedRequestIdsToRemove.Contains(response.RequestId),
                    _ => true,
                })
                .ToList();

            if (contents.Count > 0)
            {
                var clone = message.Clone();
                clone.Contents = contents;
                rewrittenMessages.Add(clone);
            }
        }

        // 子 Agent 的最终文本就是父 Agent 此次函数调用的结果。
        rewrittenMessages.Add(new ChatMessage(
            ChatRole.Tool,
            [
                new FunctionResultContent(
                    continuation.ParentCallId!,
                    childResponse.Text),
            ]));

        return new ChildResumeResult(ParentMessages: rewrittenMessages);
    }

    private static async Task<AgentResponse> RunChildAgentAsync(AIAgent agent, AgentSession session,
        IReadOnlyList<ToolApprovalResponseContent> approvalResponses, CancellationToken cancellationToken)
    {
        var approvalContents = approvalResponses
            .Cast<AIContent>()
            .ToList();

        return await agent.RunAsync([new ChatMessage(ChatRole.User, approvalContents)], session, cancellationToken: cancellationToken)
            .ConfigureAwait(false);
    }

    /// <summary>
    /// 对单条ChatMessage处理，查询content中的marker中的call id并且将其替换为sub agent待审批的请求
    /// </summary>
    /// <param name="contents">单条ChatMessage中的contents</param>
    /// <param name="parentSession">parent agent session</param>
    /// <returns>全新构造的ChatMessage的Contents</returns>
    private static List<AIContent>? TryReplaceContinuationMarkers(IList<AIContent> contents, AgentSession parentSession)
    {
        List<AIContent>? rewrittenContents = null;
        for (int i = 0; i < contents.Count; i++)
        {
            var content = contents[i];
            // 检查当前 content 是否为marker
            if (TryGetContinuationId(content, out var continuationId))
            {
                parentSession.StateBag.TryGetValue<Dictionary<string, AgentAsFunctionContinuation>>(
                    AgentAsFunctionContinuation.StateBagKey,
                    out var continuations,
                    AgentJsonUtilities.DefaultOptions);

                // 获取 call id 对应的 sub agent 数据
                if (continuations is not null && continuations.TryGetValue(continuationId, out var continuation))
                {
                    rewrittenContents ??= [.. contents.Take(i)];
                    rewrittenContents.AddRange(continuation.PendingToolApprovalRequestDict.Values);
                    continue;
                }
            }
            rewrittenContents?.Add(content);
        }
        return rewrittenContents;
    }

    private sealed record ChildResumeResult(
        List<ChatMessage>? ParentMessages = null,
        ChatResponse? ApprovalRequest = null);
}
