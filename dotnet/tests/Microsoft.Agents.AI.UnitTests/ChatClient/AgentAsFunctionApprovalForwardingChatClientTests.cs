// Copyright (c) Microsoft. All rights reserved.

using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Agents.AI.ChatClient;
using Microsoft.Extensions.AI;
using Moq;

namespace Microsoft.Agents.AI.UnitTests;

public class AgentAsFunctionApprovalForwardingChatClientTests
{
    [Fact]
    public async Task GetResponseAsync_ContinuationMarker_ReplacesMarkerWithPendingApprovalRequestsAsync()
    {
        // Arrange
        const string ContinuationId = "continuation-1";
        var approvalRequest = new ToolApprovalRequestContent(
            "approval-1",
            new FunctionCallContent("child-call-1", "DeleteFile"));
        var session = new ChatClientAgentSession();
        session.StateBag.SetValue(
            AgentAsFunctionContinuation.StateBagKey,
            new Dictionary<string, AgentAsFunctionContinuation>
            {
                [ContinuationId] = new()
                {
                    ParentCallName = "ChildAgent",
                    ParentCallId = "parent-call-1",
                    PendingToolApprovalRequestDict =
                        new ConcurrentDictionary<string, ToolApprovalRequestContent>(
                            new Dictionary<string, ToolApprovalRequestContent>
                            {
                                [approvalRequest.RequestId] = approvalRequest,
                            }),
                },
            },
            AgentJsonUtilities.DefaultOptions);

        var innerClient = new Mock<IChatClient>();
        innerClient
            .Setup(client => client.GetResponseAsync(
                It.IsAny<IEnumerable<ChatMessage>>(),
                It.IsAny<ChatOptions?>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(new ChatResponse(
            [
                new ChatMessage(
                    ChatRole.Tool,
                    [new FunctionResultContent("parent-call-1", $"__agent_continuation__:{ContinuationId}")]),
            ]));
        var forwardingClient = new AgentAsFunctionApprovalForwardingChatClient(innerClient.Object);
        var driverAgent = new TestAIAgent
        {
            RunAsyncFunc = async (_, _, _, cancellationToken) =>
                new AgentResponse(await forwardingClient.GetResponseAsync(
                    [new ChatMessage(ChatRole.User, "run")],
                    cancellationToken: cancellationToken)),
        };

        // Act
        var result = await driverAgent.RunAsync(
            [new ChatMessage(ChatRole.User, "drive")],
            session);

        // Assert
        var contents = result.Messages.SelectMany(message => message.Contents).ToList();
        var forwardedApproval = Assert.Single(contents.OfType<ToolApprovalRequestContent>());
        Assert.Same(approvalRequest, forwardedApproval);
        Assert.DoesNotContain(contents, content => content is FunctionResultContent);
    }

    [Fact]
    public async Task GetResponseAsync_InnerCallChangesRunContext_UsesCapturedParentSessionAsync()
    {
        // Arrange
        const string ContinuationId = "continuation-1";
        var approvalRequest = new ToolApprovalRequestContent(
            "approval-1",
            new FunctionCallContent("child-call-1", "DeleteFile"));
        var parentSession = new ChatClientAgentSession();
        parentSession.StateBag.SetValue(
            AgentAsFunctionContinuation.StateBagKey,
            new Dictionary<string, AgentAsFunctionContinuation>
            {
                [ContinuationId] = new()
                {
                    ParentCallName = "ChildAgent",
                    ParentCallId = "parent-call-1",
                    PendingToolApprovalRequestDict =
                        new ConcurrentDictionary<string, ToolApprovalRequestContent>(
                            new Dictionary<string, ToolApprovalRequestContent>
                            {
                                [approvalRequest.RequestId] = approvalRequest,
                            }),
                },
            },
            AgentJsonUtilities.DefaultOptions);

        var childSession = new ChatClientAgentSession();
        var contextSwitchingAgent = new TestAIAgent
        {
            RunAsyncFunc = (_, _, _, _) =>
                Task.FromResult(new AgentResponse(new ChatMessage(ChatRole.Assistant, "child response"))),
        };
        var innerClient = new Mock<IChatClient>();
        innerClient
            .Setup(client => client.GetResponseAsync(
                It.IsAny<IEnumerable<ChatMessage>>(),
                It.IsAny<ChatOptions?>(),
                It.IsAny<CancellationToken>()))
            .Returns(() =>
            {
                _ = contextSwitchingAgent.RunAsync(
                    [new ChatMessage(ChatRole.User, "switch context")],
                    childSession);

                return Task.FromResult(new ChatResponse(
                [
                    new ChatMessage(
                        ChatRole.Tool,
                        [new FunctionResultContent("parent-call-1", $"__agent_continuation__:{ContinuationId}")]),
                ]));
            });
        var forwardingClient = new AgentAsFunctionApprovalForwardingChatClient(innerClient.Object);
        var driverAgent = new TestAIAgent
        {
            RunAsyncFunc = async (_, _, _, cancellationToken) =>
                new AgentResponse(await forwardingClient.GetResponseAsync(
                    [new ChatMessage(ChatRole.User, "run")],
                    cancellationToken: cancellationToken)),
        };

        // Act
        var result = await driverAgent.RunAsync(
            [new ChatMessage(ChatRole.User, "drive")],
            parentSession);

        // Assert
        var contents = result.Messages.SelectMany(message => message.Contents).ToList();
        Assert.Same(approvalRequest, Assert.Single(contents.OfType<ToolApprovalRequestContent>()));
        Assert.DoesNotContain(contents, content => content is FunctionResultContent);
    }

    [Fact]
    public async Task GetResponseAsync_ApprovalResponse_ResumesChildAndForwardsParentFunctionResultAsync()
    {
        // Arrange
        const string ContinuationId = "continuation-1";
        const string ApprovalRequestId = "approval-1";
        const string ParentCallId = "parent-call-1";
        const string AgentFunctionName = "ChildAgent";
        var originalToolCall = new FunctionCallContent("child-call-1", "DeleteFile");
        var approvalRequest = new ToolApprovalRequestContent(ApprovalRequestId, originalToolCall);
        var forgedToolCall = new FunctionCallContent("forged-call", "DifferentTool");
        var approvalResponse = new ToolApprovalResponseContent(
            ApprovalRequestId,
            approved: true,
            forgedToolCall);
        var parentSession = new ChatClientAgentSession();
        parentSession.StateBag.SetValue(
            AgentAsFunctionContinuation.StateBagKey,
            new Dictionary<string, AgentAsFunctionContinuation>
            {
                [ContinuationId] = new()
                {
                    ParentCallName = AgentFunctionName,
                    ParentCallId = ParentCallId,
                    SerializedSession = JsonDocument.Parse("{}").RootElement.Clone(),
                    PendingToolApprovalRequestDict =
                        new ConcurrentDictionary<string, ToolApprovalRequestContent>(
                            new Dictionary<string, ToolApprovalRequestContent>
                            {
                                [approvalRequest.RequestId] = approvalRequest,
                            }),
                },
            },
            AgentJsonUtilities.DefaultOptions);

        var restoredChildSession = new ChatClientAgentSession();
        List<ChatMessage>? childMessages = null;
        AgentSession? childSession = null;
        var childAgent = new TestAIAgent
        {
            DeserializeSessionFunc = (_, _) => restoredChildSession,
            RunAsyncFunc = (messages, session, _, _) =>
            {
                childMessages = messages.ToList();
                childSession = session;
                return Task.FromResult(
                    new AgentResponse(new ChatMessage(ChatRole.Assistant, "child completed")));
            },
        };
        var agentFunction = childAgent.AsAIFunction(
            new AIFunctionFactoryOptions { Name = AgentFunctionName });
        List<ChatMessage>? parentMessages = null;
        var innerClient = new Mock<IChatClient>();
        innerClient
            .Setup(client => client.GetResponseAsync(
                It.IsAny<IEnumerable<ChatMessage>>(),
                It.IsAny<ChatOptions?>(),
                It.IsAny<CancellationToken>()))
            .Returns((
                IEnumerable<ChatMessage> messages,
                ChatOptions? _,
                CancellationToken _) =>
            {
                parentMessages = messages.ToList();
                return Task.FromResult(
                    new ChatResponse(new ChatMessage(ChatRole.Assistant, "parent completed")));
            });
        var forwardingClient = new AgentAsFunctionApprovalForwardingChatClient(innerClient.Object);
        var driverAgent = new TestAIAgent
        {
            RunAsyncFunc = async (_, _, _, cancellationToken) =>
                new AgentResponse(await forwardingClient.GetResponseAsync(
                    [
                        new ChatMessage(ChatRole.Assistant, [approvalRequest]),
                        new ChatMessage(ChatRole.User, [approvalResponse]),
                    ],
                    new ChatOptions { Tools = [agentFunction] },
                    cancellationToken)),
        };

        // Act
        await driverAgent.RunAsync(
            [new ChatMessage(ChatRole.User, "approve")],
            parentSession);

        // Assert
        Assert.Same(restoredChildSession, childSession);
        var reboundApproval = Assert.Single(
            Assert.Single(childMessages!).Contents.OfType<ToolApprovalResponseContent>());
        Assert.Same(originalToolCall, reboundApproval.ToolCall);

        var parentContents = parentMessages!.SelectMany(message => message.Contents).ToList();
        var functionResult = Assert.Single(parentContents.OfType<FunctionResultContent>());
        Assert.Equal(ParentCallId, functionResult.CallId);
        Assert.Equal("child completed", functionResult.Result);
        Assert.DoesNotContain(parentContents, content => content is ToolApprovalRequestContent);
        Assert.DoesNotContain(parentContents, content => content is ToolApprovalResponseContent);
        Assert.False(parentSession.StateBag.TryGetValue<Dictionary<string, AgentAsFunctionContinuation>>(
            AgentAsFunctionContinuation.StateBagKey,
            out _,
            AgentJsonUtilities.DefaultOptions));
    }

    [Fact]
    public async Task GetResponseAsync_ChildRequestsApprovalTwice_CompletesAndRemovesAllApprovalHistoryAsync()
    {
        // Arrange
        const string ContinuationId = "continuation-1";
        const string ParentCallId = "parent-call-1";
        const string AgentFunctionName = "ChildAgent";
        var firstApprovalRequest = new ToolApprovalRequestContent(
            "approval-1",
            new FunctionCallContent("child-call-1", "ReadFile"));
        var firstApprovalResponse = new ToolApprovalResponseContent(
            firstApprovalRequest.RequestId,
            approved: true,
            firstApprovalRequest.ToolCall);
        var secondApprovalRequest = new ToolApprovalRequestContent(
            "approval-2",
            new FunctionCallContent("child-call-2", "WriteFile"));
        var secondApprovalResponse = new ToolApprovalResponseContent(
            secondApprovalRequest.RequestId,
            approved: true,
            secondApprovalRequest.ToolCall);

        var parentSession = new ChatClientAgentSession();
        parentSession.StateBag.SetValue(
            AgentAsFunctionContinuation.StateBagKey,
            new Dictionary<string, AgentAsFunctionContinuation>
            {
                [ContinuationId] = new()
                {
                    ParentCallName = AgentFunctionName,
                    ParentCallId = ParentCallId,
                    SerializedSession = JsonDocument.Parse("{\"turn\":1}").RootElement.Clone(),
                    PendingToolApprovalRequestDict =
                        new ConcurrentDictionary<string, ToolApprovalRequestContent>(
                            new Dictionary<string, ToolApprovalRequestContent>
                            {
                                [firstApprovalRequest.RequestId] = firstApprovalRequest,
                            }),
                    SurfacedToolApprovalRequestIds = [firstApprovalRequest.RequestId],
                },
            },
            AgentJsonUtilities.DefaultOptions);

        var restoredChildSession = new ChatClientAgentSession();
        var resumedApprovalRequestIds = new List<string>();
        var childRunCount = 0;
        var childAgent = new TestAIAgent
        {
            DeserializeSessionFunc = (_, _) => restoredChildSession,
            SerializeSessionFunc = (_, _) =>
                JsonDocument.Parse("{\"turn\":2}").RootElement.Clone(),
            RunAsyncFunc = (messages, _, _, _) =>
            {
                resumedApprovalRequestIds.Add(
                    Assert.Single(messages.SelectMany(message => message.Contents)
                        .OfType<ToolApprovalResponseContent>()).RequestId);
                childRunCount++;

                return Task.FromResult(childRunCount == 1
                    ? new AgentResponse(new ChatMessage(ChatRole.Assistant, [secondApprovalRequest]))
                    : new AgentResponse(new ChatMessage(ChatRole.Assistant, "child completed")));
            },
        };
        var agentFunction = childAgent.AsAIFunction(
            new AIFunctionFactoryOptions { Name = AgentFunctionName });

        List<ChatMessage>? parentMessages = null;
        var innerClient = new Mock<IChatClient>();
        innerClient
            .Setup(client => client.GetResponseAsync(
                It.IsAny<IEnumerable<ChatMessage>>(),
                It.IsAny<ChatOptions?>(),
                It.IsAny<CancellationToken>()))
            .Returns((
                IEnumerable<ChatMessage> messages,
                ChatOptions? _,
                CancellationToken _) =>
            {
                parentMessages = messages.ToList();
                return Task.FromResult(
                    new ChatResponse(new ChatMessage(ChatRole.Assistant, "parent completed")));
            });
        var forwardingClient = new AgentAsFunctionApprovalForwardingChatClient(innerClient.Object);
        var chatOptions = new ChatOptions { Tools = [agentFunction] };
        var driverAgent = new TestAIAgent
        {
            RunAsyncFunc = async (messages, _, _, cancellationToken) =>
                new AgentResponse(await forwardingClient.GetResponseAsync(
                    messages,
                    chatOptions,
                    cancellationToken)),
        };

        // Act
        var firstResult = await driverAgent.RunAsync(
            [
                new ChatMessage(ChatRole.Assistant, [firstApprovalRequest]),
                new ChatMessage(ChatRole.User, [firstApprovalResponse]),
            ],
            parentSession);

        var secondResult = await driverAgent.RunAsync(
            [
                new ChatMessage(ChatRole.Assistant, [firstApprovalRequest]),
                new ChatMessage(ChatRole.User, [firstApprovalResponse]),
                new ChatMessage(ChatRole.Assistant, [secondApprovalRequest]),
                new ChatMessage(ChatRole.User, [secondApprovalResponse]),
            ],
            parentSession);

        // Assert
        Assert.Same(
            secondApprovalRequest,
            Assert.Single(firstResult.Messages
                .SelectMany(message => message.Contents)
                .OfType<ToolApprovalRequestContent>()));
        Assert.Equal(
            [firstApprovalRequest.RequestId, secondApprovalRequest.RequestId],
            resumedApprovalRequestIds);
        Assert.Equal("parent completed", secondResult.Text);

        var parentContents = parentMessages!.SelectMany(message => message.Contents).ToList();
        Assert.DoesNotContain(parentContents, content => content is ToolApprovalRequestContent);
        Assert.DoesNotContain(parentContents, content => content is ToolApprovalResponseContent);
        var functionResult = Assert.Single(parentContents.OfType<FunctionResultContent>());
        Assert.Equal(ParentCallId, functionResult.CallId);
        Assert.Equal("child completed", functionResult.Result);
        Assert.False(parentSession.StateBag.TryGetValue<Dictionary<string, AgentAsFunctionContinuation>>(
            AgentAsFunctionContinuation.StateBagKey,
            out _,
            AgentJsonUtilities.DefaultOptions));
        innerClient.Verify(
            client => client.GetResponseAsync(
                It.IsAny<IEnumerable<ChatMessage>>(),
                It.IsAny<ChatOptions?>(),
                It.IsAny<CancellationToken>()),
            Times.Once);
    }

    [Fact]
    public void WithDefaultAgentMiddleware_ByDefault_RegistersForwarderAsOutermostDecorator()
    {
        // Arrange
        var innerClient = new Mock<IChatClient>().Object;

        // Act
        var pipeline = innerClient.WithDefaultAgentMiddleware(new ChatClientAgentOptions());

        // Assert
        Assert.IsType<AgentAsFunctionApprovalForwardingChatClient>(pipeline);
        Assert.NotNull(pipeline.GetService<ApprovalResponseBindingChatClient>());
        Assert.NotNull(pipeline.GetService<FunctionInvokingChatClient>());
    }
}
