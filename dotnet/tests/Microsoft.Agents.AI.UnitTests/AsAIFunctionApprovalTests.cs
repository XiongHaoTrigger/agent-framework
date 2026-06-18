// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Moq;

namespace Microsoft.Agents.AI.UnitTests;

/// <summary>
/// Unit tests for agent-as-function approval propagation.
/// </summary>
public class AsAIFunctionApprovalTests
{
    [Fact]
    public async Task AsAIFunction_AsParentTool_SurfacesChildApprovalRequestAsync()
    {
        // Arrange
        var childToolCall = new FunctionCallContent("childCall1", "DangerousChildTool");
        var childApprovalRequest = new ToolApprovalRequestContent("childRequest1", childToolCall);
        var childAgent = new ApprovalTestAgent(new AgentResponse(new ChatMessage(ChatRole.Assistant, [childApprovalRequest])));
        var childAsTool = childAgent.AsAIFunction();
        var parentToolCall = CreateParentToolCall("parentCall1", childAsTool);

        // Act
        var result = await ChatClientAgentTestHelper.RunAsync(
            inputMessages: [new(ChatRole.User, "Delegate to the child agent")],
            serviceCallExpectations:
            [
                new(new ChatResponse([new(ChatRole.Assistant, [parentToolCall])]))
            ],
            agentOptions: new()
            {
                ChatOptions = new() { Tools = [childAsTool] },
            },
            expectedServiceCallCount: 1);

        // Assert
        ToolApprovalRequestContent surfacedApproval = GetSingleApprovalRequest(result.Response.Messages);
        Assert.Equal("childRequest1", surfacedApproval.RequestId);
        Assert.Same(childToolCall, surfacedApproval.ToolCall);
        AssertParentCallPrecedesApproval(result.Response.Messages, parentToolCall, surfacedApproval);
        Assert.Equal(1, childAgent.RunAsyncCallCount);
        Assert.DoesNotContain(result.Response.Messages.SelectMany(static m => m.Contents), IsPendingApprovalMarker);
    }

    [Fact]
    public async Task AsAIFunction_AsParentTool_WithApprovalResponse_ResumesChildAgentAsync()
    {
        // Arrange
        const string FinalChildText = "Child action completed.";
        var childToolCall = new FunctionCallContent("childCall1", "DangerousChildTool");
        var childApprovalRequest = new ToolApprovalRequestContent("childRequest1", childToolCall);
        var childAgent = new ApprovalTestAgent(
            new AgentResponse(new ChatMessage(ChatRole.Assistant, [childApprovalRequest])),
            new AgentResponse(new ChatMessage(ChatRole.Assistant, FinalChildText)));
        var childAsTool = childAgent.AsAIFunction();
        var parentToolCall = CreateParentToolCall("parentCall1", childAsTool);
        var callIndex = new ChatClientAgentTestHelper.Ref<int>(0);
        var capturedInputs = new List<List<ChatMessage>>();
        var serviceCallExpectations = new List<ChatClientAgentTestHelper.ServiceCallExpectation>
        {
            new(new ChatResponse([new(ChatRole.Assistant, [parentToolCall])])),
            new(
                new ChatResponse([new(ChatRole.Assistant, "Parent observed child completion.")]),
                messages => AssertFunctionResult(messages, "parentCall1", FinalChildText)),
        };

        var firstRun = await ChatClientAgentTestHelper.RunAsync(
            inputMessages: [new(ChatRole.User, "Delegate to the child agent")],
            serviceCallExpectations: serviceCallExpectations,
            agentOptions: new()
            {
                ChatOptions = new() { Tools = [childAsTool] },
            },
            callIndex: callIndex,
            capturedInputs: capturedInputs);

        ToolApprovalRequestContent surfacedApproval = GetSingleApprovalRequest(firstRun.Response.Messages);

        // Act
        var secondRun = await ChatClientAgentTestHelper.RunAsync(
            inputMessages: [new(ChatRole.User, [surfacedApproval.CreateResponse(approved: true)])],
            serviceCallExpectations: serviceCallExpectations,
            existingSession: firstRun.Session,
            existingAgent: firstRun.Agent,
            existingMock: firstRun.MockService,
            callIndex: callIndex,
            capturedInputs: capturedInputs,
            expectedServiceCallCount: 2);

        // Assert
        Assert.Equal(2, childAgent.RunAsyncCallCount);
        Assert.Equal("Parent observed child completion.", secondRun.Response.Text);
    }

    [Fact]
    public async Task AsAIFunction_AsParentTool_WithSequentialApprovals_SurfacesNextApprovalBeforeParentContinuesAsync()
    {
        // Arrange
        const string FinalChildText = "Child action completed after approvals.";
        var firstChildToolCall = new FunctionCallContent("childCall1", "FirstDangerousChildTool");
        var firstChildApprovalRequest = new ToolApprovalRequestContent("childRequest1", firstChildToolCall);
        var secondChildToolCall = new FunctionCallContent("childCall2", "SecondDangerousChildTool");
        var secondChildApprovalRequest = new ToolApprovalRequestContent("childRequest2", secondChildToolCall);
        var childAgent = new ApprovalTestAgent(
            new AgentResponse(new ChatMessage(ChatRole.Assistant, [firstChildApprovalRequest])),
            new AgentResponse(new ChatMessage(ChatRole.Assistant, [secondChildApprovalRequest])),
            new AgentResponse(new ChatMessage(ChatRole.Assistant, FinalChildText)));
        var childAsTool = childAgent.AsAIFunction();
        var parentToolCall = CreateParentToolCall("parentCall1", childAsTool);
        var callIndex = new ChatClientAgentTestHelper.Ref<int>(0);
        var capturedInputs = new List<List<ChatMessage>>();
        var serviceCallExpectations = new List<ChatClientAgentTestHelper.ServiceCallExpectation>
        {
            new(new ChatResponse([new(ChatRole.Assistant, [parentToolCall])])),
            new(
                new ChatResponse([new(ChatRole.Assistant, "Parent observed child completion.")]),
                messages => AssertFunctionResult(messages, "parentCall1", FinalChildText)),
        };

        var firstRun = await ChatClientAgentTestHelper.RunAsync(
            inputMessages: [new(ChatRole.User, "Delegate to the child agent")],
            serviceCallExpectations: serviceCallExpectations,
            agentOptions: new()
            {
                ChatOptions = new() { Tools = [childAsTool] },
            },
            callIndex: callIndex,
            capturedInputs: capturedInputs);

        ToolApprovalRequestContent firstApproval = GetSingleApprovalRequest(firstRun.Response.Messages);

        var secondRun = await ChatClientAgentTestHelper.RunAsync(
            inputMessages: [new(ChatRole.User, [firstApproval.CreateResponse(approved: true)])],
            serviceCallExpectations: serviceCallExpectations,
            existingSession: firstRun.Session,
            existingAgent: firstRun.Agent,
            existingMock: firstRun.MockService,
            callIndex: callIndex,
            capturedInputs: capturedInputs,
            expectedServiceCallCount: 1);

        ToolApprovalRequestContent secondApproval = GetSingleApprovalRequest(secondRun.Response.Messages);

        // Act
        var thirdRun = await ChatClientAgentTestHelper.RunAsync(
            inputMessages: [new(ChatRole.User, [secondApproval.CreateResponse(approved: true)])],
            serviceCallExpectations: serviceCallExpectations,
            existingSession: secondRun.Session,
            existingAgent: secondRun.Agent,
            existingMock: secondRun.MockService,
            callIndex: callIndex,
            capturedInputs: capturedInputs,
            expectedServiceCallCount: 2);

        // Assert
        Assert.Equal("childRequest1", firstApproval.RequestId);
        Assert.Equal("childRequest2", secondApproval.RequestId);
        Assert.Same(secondChildToolCall, secondApproval.ToolCall);
        Assert.Equal(3, childAgent.RunAsyncCallCount);
        Assert.Equal("Parent observed child completion.", thirdRun.Response.Text);
    }

    [Fact]
    public async Task AsAIFunction_NestedAgents_WithApprovalResponse_ResumesFullChainAsync()
    {
        // Arrange
        const string FinalLeafText = "Leaf action completed.";
        const string FinalMiddleText = "Middle observed leaf completion.";
        const string FinalParentText = "Parent observed middle completion.";
        var leafToolCall = new FunctionCallContent("leafCall1", "DangerousLeafTool");
        var leafApprovalRequest = new ToolApprovalRequestContent("leafRequest1", leafToolCall);
        var leafAgent = new ApprovalTestAgent(
            new AgentResponse(new ChatMessage(ChatRole.Assistant, [leafApprovalRequest])),
            new AgentResponse(new ChatMessage(ChatRole.Assistant, FinalLeafText)));
        var leafAsTool = leafAgent.AsAIFunction();
        var middleToolCall = CreateParentToolCall("middleCall1", leafAsTool);
        var middleCallIndex = new ChatClientAgentTestHelper.Ref<int>(0);
        var middleCapturedInputs = new List<List<ChatMessage>>();
        var middleMock = ChatClientAgentTestHelper.CreateSequentialMock(
            [
                new(new ChatResponse([new(ChatRole.Assistant, [middleToolCall])])),
                new(
                    new ChatResponse([new(ChatRole.Assistant, FinalMiddleText)]),
                    messages => AssertFunctionResult(messages, "middleCall1", FinalLeafText)),
            ],
            middleCallIndex,
            middleCapturedInputs);
        var middleAgent = new ChatClientAgent(
            middleMock.Object,
            options: new()
            {
                ChatOptions = new() { Tools = [leafAsTool] },
            },
            services: new ServiceCollection().BuildServiceProvider());
        var middleSession = (await middleAgent.CreateSessionAsync()) as ChatClientAgentSession;
        var middleAsTool = middleAgent.AsAIFunction(session: middleSession);
        var parentToolCall = CreateParentToolCall("parentCall1", middleAsTool);
        var parentCallIndex = new ChatClientAgentTestHelper.Ref<int>(0);
        var parentCapturedInputs = new List<List<ChatMessage>>();
        var parentExpectations = new List<ChatClientAgentTestHelper.ServiceCallExpectation>
        {
            new(new ChatResponse([new(ChatRole.Assistant, [parentToolCall])])),
            new(
                new ChatResponse([new(ChatRole.Assistant, FinalParentText)]),
                messages => AssertFunctionResult(messages, "parentCall1", FinalMiddleText)),
        };

        var firstRun = await ChatClientAgentTestHelper.RunAsync(
            inputMessages: [new(ChatRole.User, "Delegate through the middle agent")],
            serviceCallExpectations: parentExpectations,
            agentOptions: new()
            {
                ChatOptions = new() { Tools = [middleAsTool] },
            },
            callIndex: parentCallIndex,
            capturedInputs: parentCapturedInputs);

        ToolApprovalRequestContent surfacedApproval = GetSingleApprovalRequest(firstRun.Response.Messages);

        // Act
        var secondRun = await ChatClientAgentTestHelper.RunAsync(
            inputMessages: [new(ChatRole.User, [surfacedApproval.CreateResponse(approved: true)])],
            serviceCallExpectations: parentExpectations,
            existingSession: firstRun.Session,
            existingAgent: firstRun.Agent,
            existingMock: firstRun.MockService,
            callIndex: parentCallIndex,
            capturedInputs: parentCapturedInputs,
            expectedServiceCallCount: 2);

        // Assert
        Assert.Equal("leafRequest1", surfacedApproval.RequestId);
        Assert.Same(leafToolCall, surfacedApproval.ToolCall);
        Assert.Equal(2, leafAgent.RunAsyncCallCount);
        Assert.Equal(2, middleCallIndex.Value);
        Assert.Equal(FinalParentText, secondRun.Response.Text);
    }

    [Fact]
    public async Task AsAIFunction_AsParentToolStreaming_SurfacesChildApprovalRequestAsync()
    {
        // Arrange
        var childToolCall = new FunctionCallContent("childCall1", "DangerousChildTool");
        var childApprovalRequest = new ToolApprovalRequestContent("childRequest1", childToolCall);
        var childAgent = new ApprovalTestAgent(new AgentResponse(new ChatMessage(ChatRole.Assistant, [childApprovalRequest])));
        var childAsTool = childAgent.AsAIFunction();
        var parentToolCall = CreateParentToolCall("parentCall1", childAsTool);
        var mock = new Mock<IChatClient>();
        mock.Setup(c => c.GetStreamingResponseAsync(
                It.IsAny<IEnumerable<ChatMessage>>(),
                It.IsAny<ChatOptions>(),
                It.IsAny<CancellationToken>()))
            .Returns(ToAsyncEnumerableAsync([new ChatResponseUpdate(ChatRole.Assistant, [parentToolCall])]));
        var agent = new ChatClientAgent(
            mock.Object,
            options: new()
            {
                ChatOptions = new() { Tools = [childAsTool] },
            },
            services: new ServiceCollection().BuildServiceProvider());
        var session = await agent.CreateSessionAsync();
        var updates = new List<AgentResponseUpdate>();

        // Act
        await foreach (var update in agent.RunStreamingAsync(
            [new ChatMessage(ChatRole.User, "Delegate to the child agent")],
            session))
        {
            updates.Add(update);
        }

        // Assert
        ToolApprovalRequestContent surfacedApproval = updates
            .SelectMany(static u => u.Contents)
            .OfType<ToolApprovalRequestContent>()
            .Single();

        Assert.Equal("childRequest1", surfacedApproval.RequestId);
        Assert.Same(childToolCall, surfacedApproval.ToolCall);
        AssertParentCallPrecedesApproval(updates, parentToolCall, surfacedApproval);
        Assert.Equal(1, childAgent.RunAsyncCallCount);
    }

    private static FunctionCallContent CreateParentToolCall(string callId, AIFunction tool)
        => new(callId, tool.Name, new Dictionary<string, object?> { ["query"] = "Run the child task" });

    private static ToolApprovalRequestContent GetSingleApprovalRequest(IEnumerable<ChatMessage> messages)
        => messages
            .SelectMany(static m => m.Contents)
            .OfType<ToolApprovalRequestContent>()
            .Single();

    private static void AssertFunctionResult(List<ChatMessage> messages, string callId, string result)
    {
        FunctionResultContent functionResult = messages
            .SelectMany(static m => m.Contents)
            .OfType<FunctionResultContent>()
            .Single(content => content.CallId == callId);

        Assert.Equal(result, functionResult.Result);
    }

    private static void AssertParentCallPrecedesApproval(
        IEnumerable<ChatMessage> messages,
        FunctionCallContent parentToolCall,
        ToolApprovalRequestContent approvalRequest)
    {
        AssertParentCallPrecedesApproval(messages.SelectMany(static m => m.Contents), parentToolCall, approvalRequest);
    }

    private static void AssertParentCallPrecedesApproval(
        IEnumerable<AgentResponseUpdate> updates,
        FunctionCallContent parentToolCall,
        ToolApprovalRequestContent approvalRequest)
    {
        AssertParentCallPrecedesApproval(updates.SelectMany(static u => u.Contents), parentToolCall, approvalRequest);
    }

    private static void AssertParentCallPrecedesApproval(
        IEnumerable<AIContent> contents,
        FunctionCallContent parentToolCall,
        ToolApprovalRequestContent approvalRequest)
    {
        var contentList = contents.ToList();
        int parentIndex = contentList.IndexOf(parentToolCall);
        int approvalIndex = contentList.IndexOf(approvalRequest);

        Assert.True(parentIndex >= 0, "Parent function call was not found.");
        Assert.True(approvalIndex >= 0, "Child approval request was not found.");
        Assert.True(parentIndex < approvalIndex, "Parent function call should precede child approval request.");
    }

    private static bool IsPendingApprovalMarker(AIContent content)
        => content is FunctionResultContent functionResult &&
            functionResult.Result?.ToString()?.StartsWith(
                AgentAsFunctionApprovalDelegatingChatClient.PendingApprovalMarkerPrefix,
                StringComparison.Ordinal) == true;

    private static async IAsyncEnumerable<ChatResponseUpdate> ToAsyncEnumerableAsync(IEnumerable<ChatResponseUpdate> updates)
    {
        foreach (var update in updates)
        {
            await Task.Yield();
            yield return update;
        }
    }

    private sealed class ApprovalTestAgent : AIAgent
    {
        private readonly AgentResponse[] _responses;

        public ApprovalTestAgent(params AgentResponse[] responses)
        {
            this._responses = responses;
        }

        public int RunAsyncCallCount { get; private set; }

        protected override ValueTask<AgentSession> CreateSessionCoreAsync(CancellationToken cancellationToken = default)
            => new(new TestAgentSession());

        protected override ValueTask<JsonElement> SerializeSessionCoreAsync(
            AgentSession session,
            JsonSerializerOptions? jsonSerializerOptions = null,
            CancellationToken cancellationToken = default)
            => throw new NotImplementedException();

        protected override ValueTask<AgentSession> DeserializeSessionCoreAsync(
            JsonElement serializedState,
            JsonSerializerOptions? jsonSerializerOptions = null,
            CancellationToken cancellationToken = default)
            => throw new NotImplementedException();

        protected override Task<AgentResponse> RunCoreAsync(
            IEnumerable<ChatMessage> messages,
            AgentSession? session = null,
            AgentRunOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            int index = this.RunAsyncCallCount++;
            if (index >= this._responses.Length)
            {
                throw new InvalidOperationException("No response configured for this child agent call.");
            }

            return Task.FromResult(this._responses[index]);
        }

        protected override async IAsyncEnumerable<AgentResponseUpdate> RunCoreStreamingAsync(
            IEnumerable<ChatMessage> messages,
            AgentSession? session = null,
            AgentRunOptions? options = null,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var response = await this.RunAsync(messages, session, options, cancellationToken).ConfigureAwait(false);
            foreach (var update in response.ToAgentResponseUpdates())
            {
                yield return update;
            }
        }
    }

    private sealed class TestAgentSession : AgentSession;
}
