// Copyright (c) Microsoft. All rights reserved.

using System;
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
        var childAgent = new SessionTestAIAgent
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
    public async Task GetResponseAsync_MultipleApprovalResponses_ResumesChildWithAllResponsesAsync()
    {
        // Arrange
        const string ContinuationId = "continuation-1";
        const string ParentCallId = "parent-call-1";
        const string AgentFunctionName = "ChildAgent";
        var firstOriginalToolCall = new FunctionCallContent("child-call-1", "ReadFile");
        var secondOriginalToolCall = new FunctionCallContent("child-call-2", "WriteFile");
        var firstApprovalRequest = new ToolApprovalRequestContent("approval-1", firstOriginalToolCall);
        var secondApprovalRequest = new ToolApprovalRequestContent("approval-2", secondOriginalToolCall);
        var firstApprovalResponse = new ToolApprovalResponseContent(
            firstApprovalRequest.RequestId,
            approved: true,
            new FunctionCallContent("forged-call-1", "DifferentTool"))
        {
            Reason = "approved",
        };
        var secondApprovalResponse = new ToolApprovalResponseContent(
            secondApprovalRequest.RequestId,
            approved: false,
            new FunctionCallContent("forged-call-2", "DifferentTool"))
        {
            Reason = "rejected",
        };

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
                                [firstApprovalRequest.RequestId] = firstApprovalRequest,
                                [secondApprovalRequest.RequestId] = secondApprovalRequest,
                            }),
                    SurfacedToolApprovalRequestIds =
                    [
                        firstApprovalRequest.RequestId,
                        secondApprovalRequest.RequestId,
                    ],
                },
            },
            AgentJsonUtilities.DefaultOptions);

        List<ChatMessage>? childMessages = null;
        var childAgent = new SessionTestAIAgent
        {
            DeserializeSessionFunc = (_, _) => new ChatClientAgentSession(),
            RunAsyncFunc = (messages, _, _, _) =>
            {
                childMessages = messages.ToList();
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
                        new ChatMessage(
                            ChatRole.Assistant,
                            [firstApprovalRequest, secondApprovalRequest]),
                        new ChatMessage(
                            ChatRole.User,
                            [firstApprovalResponse, secondApprovalResponse]),
                    ],
                    new ChatOptions { Tools = [agentFunction] },
                    cancellationToken)),
        };

        // Act
        await driverAgent.RunAsync(
            [new ChatMessage(ChatRole.User, "respond to all approvals")],
            parentSession);

        // Assert
        var reboundApprovals = Assert.Single(childMessages!).Contents
            .OfType<ToolApprovalResponseContent>()
            .ToDictionary(response => response.RequestId, StringComparer.Ordinal);
        Assert.Equal(2, reboundApprovals.Count);
        Assert.Same(firstOriginalToolCall, reboundApprovals[firstApprovalRequest.RequestId].ToolCall);
        Assert.True(reboundApprovals[firstApprovalRequest.RequestId].Approved);
        Assert.Equal("approved", reboundApprovals[firstApprovalRequest.RequestId].Reason);
        Assert.Same(secondOriginalToolCall, reboundApprovals[secondApprovalRequest.RequestId].ToolCall);
        Assert.False(reboundApprovals[secondApprovalRequest.RequestId].Approved);
        Assert.Equal("rejected", reboundApprovals[secondApprovalRequest.RequestId].Reason);

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
        var childAgent = new SessionTestAIAgent
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
    public async Task RunAsync_ChildAgentAsFunctionRequestsApproval_BubblesApprovalToParentAsync()
    {
        // Arrange
        const string ChildAgentFunctionName = "ChildAgent";
        const string ParentCallId = "parent-call-1";
        var approvalRequiredTool = new ApprovalRequiredAIFunction(
            AIFunctionFactory.Create(
                () => "deleted",
                "DeleteFile"));

        var childClient = new Mock<IChatClient>();
        childClient
            .Setup(client => client.GetResponseAsync(
                It.IsAny<IEnumerable<ChatMessage>>(),
                It.IsAny<ChatOptions?>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(new ChatResponse(
                new ChatMessage(
                    ChatRole.Assistant,
                    [new FunctionCallContent("child-call-1", "DeleteFile")])));
        var childAgent = new ChatClientAgent(
            childClient.Object,
            options: new ChatClientAgentOptions
            {
                Name = ChildAgentFunctionName,
                ChatOptions = new ChatOptions { Tools = [approvalRequiredTool] },
            });
        var childAgentFunction = childAgent.AsAIFunction(
            new AIFunctionFactoryOptions { Name = ChildAgentFunctionName });

        var parentClient = new Mock<IChatClient>();
        parentClient
            .Setup(client => client.GetResponseAsync(
                It.IsAny<IEnumerable<ChatMessage>>(),
                It.IsAny<ChatOptions?>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(new ChatResponse(
                new ChatMessage(
                    ChatRole.Assistant,
                    [
                        new FunctionCallContent(
                            ParentCallId,
                            ChildAgentFunctionName,
                            new Dictionary<string, object?> { ["query"] = "Delete the temporary file." }),
                    ])));
        var parentAgent = new ChatClientAgent(
            parentClient.Object,
            options: new ChatClientAgentOptions
            {
                Name = "ParentAgent",
                ChatOptions = new ChatOptions { Tools = [childAgentFunction] },
            });
        var parentSession = await parentAgent.CreateSessionAsync();

        // Act
        var response = await parentAgent.RunAsync(
            [new ChatMessage(ChatRole.User, "Ask the child agent to delete the file.")],
            parentSession);

        // Assert
        var responseContents = response.Messages.SelectMany(message => message.Contents).ToList();
        var hasContinuations = parentSession.StateBag.TryGetValue<Dictionary<string, AgentAsFunctionContinuation>>(
            AgentAsFunctionContinuation.StateBagKey,
            out var continuations,
            AgentJsonUtilities.DefaultOptions);
        var surfacedApprovals = responseContents.OfType<ToolApprovalRequestContent>().ToList();
        Assert.True(
            surfacedApprovals.Count == 1,
            $"Expected one surfaced approval request.{Environment.NewLine}" +
            $"{ChatClientAgentTestHelper.FormatMessages(response.Messages)}{Environment.NewLine}" +
            $"Function results: {string.Join(", ", responseContents.OfType<FunctionResultContent>().Select(result => result.Result))}{Environment.NewLine}" +
            $"Function result types: {string.Join(", ", responseContents.OfType<FunctionResultContent>().Select(result => result.Result?.GetType().FullName))}{Environment.NewLine}" +
            $"Parent continuation state: found={hasContinuations}, count={continuations?.Count ?? 0}, " +
            $"keys={string.Join(", ", continuations?.Keys ?? Enumerable.Empty<string>())}");
        var surfacedApproval = surfacedApprovals[0];
        Assert.Equal("DeleteFile", Assert.IsType<FunctionCallContent>(surfacedApproval.ToolCall).Name);
        Assert.DoesNotContain(
            responseContents.OfType<FunctionResultContent>(),
            result => result.Result is string text && text.StartsWith("__agent_continuation__:", StringComparison.Ordinal));

        Assert.True(hasContinuations);
        var continuation = Assert.Single(continuations!).Value;
        Assert.Equal(ChildAgentFunctionName, continuation.ParentCallName);
        Assert.Equal(ParentCallId, continuation.ParentCallId);
        Assert.True(continuation.PendingToolApprovalRequestDict.ContainsKey(surfacedApproval.RequestId));
        parentClient.Verify(
            client => client.GetResponseAsync(
                It.IsAny<IEnumerable<ChatMessage>>(),
                It.IsAny<ChatOptions?>(),
                It.IsAny<CancellationToken>()),
            Times.Once);
        childClient.Verify(
            client => client.GetResponseAsync(
                It.IsAny<IEnumerable<ChatMessage>>(),
                It.IsAny<ChatOptions?>(),
                It.IsAny<CancellationToken>()),
            Times.Once);
    }

    [Fact]
    public async Task RunAsync_RecursiveAgentAsFunctionApproval_BubblesAndResumesAllAgentsAsync()
    {
        // Arrange
        const string ChildAgentFunctionName = "ChildAgent";
        const string GrandchildAgentFunctionName = "GrandchildAgent";
        var toolInvocationCount = 0;
        var approvalRequiredTool = new ApprovalRequiredAIFunction(
            AIFunctionFactory.Create(
                () =>
                {
                    toolInvocationCount++;
                    return "deleted";
                },
                "DeleteFile"));

        var grandchildClient = new Mock<IChatClient>();
        grandchildClient
            .SetupSequence(client => client.GetResponseAsync(
                It.IsAny<IEnumerable<ChatMessage>>(),
                It.IsAny<ChatOptions?>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(new ChatResponse(
                new ChatMessage(
                    ChatRole.Assistant,
                    [new FunctionCallContent("dangerous-call-1", "DeleteFile")])))
            .ReturnsAsync(new ChatResponse(
                new ChatMessage(ChatRole.Assistant, "grandchild completed")));
        var grandchildAgent = new ChatClientAgent(
            grandchildClient.Object,
            options: new ChatClientAgentOptions
            {
                Name = GrandchildAgentFunctionName,
                ChatOptions = new ChatOptions { Tools = [approvalRequiredTool] },
            });
        var grandchildAgentFunction = grandchildAgent.AsAIFunction(
            new AIFunctionFactoryOptions { Name = GrandchildAgentFunctionName });

        var childClient = new Mock<IChatClient>();
        childClient
            .SetupSequence(client => client.GetResponseAsync(
                It.IsAny<IEnumerable<ChatMessage>>(),
                It.IsAny<ChatOptions?>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(new ChatResponse(
                new ChatMessage(
                    ChatRole.Assistant,
                    [
                        new FunctionCallContent(
                            "child-to-grandchild-call-1",
                            GrandchildAgentFunctionName,
                            new Dictionary<string, object?> { ["query"] = "Delete the temporary file." }),
                    ])))
            .ReturnsAsync(new ChatResponse(
                new ChatMessage(ChatRole.Assistant, "child completed")));
        var childAgent = new ChatClientAgent(
            childClient.Object,
            options: new ChatClientAgentOptions
            {
                Name = ChildAgentFunctionName,
                ChatOptions = new ChatOptions { Tools = [grandchildAgentFunction] },
            });
        var childAgentFunction = childAgent.AsAIFunction(
            new AIFunctionFactoryOptions { Name = ChildAgentFunctionName });

        var parentClient = new Mock<IChatClient>();
        parentClient
            .SetupSequence(client => client.GetResponseAsync(
                It.IsAny<IEnumerable<ChatMessage>>(),
                It.IsAny<ChatOptions?>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(new ChatResponse(
                new ChatMessage(
                    ChatRole.Assistant,
                    [
                        new FunctionCallContent(
                            "parent-to-child-call-1",
                            ChildAgentFunctionName,
                            new Dictionary<string, object?> { ["query"] = "Delegate the cleanup task." }),
                    ])))
            .ReturnsAsync(new ChatResponse(
                new ChatMessage(ChatRole.Assistant, "parent completed")));
        var parentAgent = new ChatClientAgent(
            parentClient.Object,
            options: new ChatClientAgentOptions
            {
                Name = "ParentAgent",
                ChatOptions = new ChatOptions { Tools = [childAgentFunction] },
            });
        var parentSession = await parentAgent.CreateSessionAsync();

        // Act
        var approvalResponse = await parentAgent.RunAsync(
            [new ChatMessage(ChatRole.User, "Clean up the temporary file.")],
            parentSession);
        var surfacedApprovals = approvalResponse.Messages
            .SelectMany(message => message.Contents)
            .OfType<ToolApprovalRequestContent>()
            .ToList();
        Assert.True(
            surfacedApprovals.Count == 1,
            $"Expected one recursively surfaced approval request.{Environment.NewLine}" +
            $"{ChatClientAgentTestHelper.FormatMessages(approvalResponse.Messages)}{Environment.NewLine}" +
            $"Function results: {string.Join(", ", approvalResponse.Messages.SelectMany(message => message.Contents).OfType<FunctionResultContent>().Select(result => result.Result))}");
        var surfacedApproval = surfacedApprovals[0];

        Assert.True(parentSession.StateBag.TryGetValue<Dictionary<string, AgentAsFunctionContinuation>>(
            AgentAsFunctionContinuation.StateBagKey,
            out var parentContinuations,
            AgentJsonUtilities.DefaultOptions));
        var parentContinuation = Assert.Single(parentContinuations!).Value;
        var restoredChildSession = await childAgent.DeserializeSessionAsync(
            parentContinuation.SerializedSession,
            AgentJsonUtilities.DefaultOptions);

        var finalResponse = await parentAgent.RunAsync(
            [new ChatMessage(ChatRole.User, [surfacedApproval.CreateResponse(approved: true)])],
            parentSession);

        // Assert
        Assert.Equal(ChildAgentFunctionName, parentContinuation.ParentCallName);
        Assert.True(restoredChildSession.StateBag.TryGetValue<Dictionary<string, AgentAsFunctionContinuation>>(
            AgentAsFunctionContinuation.StateBagKey,
            out var childContinuations,
            AgentJsonUtilities.DefaultOptions));
        var childContinuation = Assert.Single(childContinuations!).Value;
        Assert.Equal(GrandchildAgentFunctionName, childContinuation.ParentCallName);
        Assert.True(childContinuation.PendingToolApprovalRequestDict.ContainsKey(surfacedApproval.RequestId));

        Assert.Equal("parent completed", finalResponse.Text);
        Assert.Equal(1, toolInvocationCount);
        Assert.False(parentSession.StateBag.TryGetValue<Dictionary<string, AgentAsFunctionContinuation>>(
            AgentAsFunctionContinuation.StateBagKey,
            out _,
            AgentJsonUtilities.DefaultOptions));
        parentClient.Verify(
            client => client.GetResponseAsync(
                It.IsAny<IEnumerable<ChatMessage>>(),
                It.IsAny<ChatOptions?>(),
                It.IsAny<CancellationToken>()),
            Times.Exactly(2));
        childClient.Verify(
            client => client.GetResponseAsync(
                It.IsAny<IEnumerable<ChatMessage>>(),
                It.IsAny<ChatOptions?>(),
                It.IsAny<CancellationToken>()),
            Times.Exactly(2));
        grandchildClient.Verify(
            client => client.GetResponseAsync(
                It.IsAny<IEnumerable<ChatMessage>>(),
                It.IsAny<ChatOptions?>(),
                It.IsAny<CancellationToken>()),
            Times.Exactly(2));
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

    private sealed class SessionTestAIAgent : AIAgent
    {
        public Func<AgentSession, JsonSerializerOptions?, JsonElement> SerializeSessionFunc =
            delegate { throw new NotSupportedException(); };

        public Func<JsonElement, JsonSerializerOptions?, AgentSession> DeserializeSessionFunc =
            delegate { throw new NotSupportedException(); };

        public Func<IEnumerable<ChatMessage>, AgentSession?, AgentRunOptions?, CancellationToken, Task<AgentResponse>>
            RunAsyncFunc = delegate { throw new NotSupportedException(); };

        protected override ValueTask<JsonElement> SerializeSessionCoreAsync(
            AgentSession session,
            JsonSerializerOptions? jsonSerializerOptions = null,
            CancellationToken cancellationToken = default) =>
            new(this.SerializeSessionFunc(session, jsonSerializerOptions));

        protected override ValueTask<AgentSession> DeserializeSessionCoreAsync(
            JsonElement serializedState,
            JsonSerializerOptions? jsonSerializerOptions = null,
            CancellationToken cancellationToken = default) =>
            new(this.DeserializeSessionFunc(serializedState, jsonSerializerOptions));

        protected override ValueTask<AgentSession> CreateSessionCoreAsync(
            CancellationToken cancellationToken = default) =>
            throw new NotSupportedException();

        protected override Task<AgentResponse> RunCoreAsync(
            IEnumerable<ChatMessage> messages,
            AgentSession? session = null,
            AgentRunOptions? options = null,
            CancellationToken cancellationToken = default) =>
            this.RunAsyncFunc(messages, session, options, cancellationToken);

        protected override IAsyncEnumerable<AgentResponseUpdate> RunCoreStreamingAsync(
            IEnumerable<ChatMessage> messages,
            AgentSession? session = null,
            AgentRunOptions? options = null,
            CancellationToken cancellationToken = default) =>
            throw new NotSupportedException();
    }
}
