// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Moq;

namespace Microsoft.Agents.AI.UnitTests;

/// <summary>
/// Contains unit tests that verify the end-to-end approval flow behavior of the
/// <see cref="ChatClientAgent"/> class with <see cref="PerServiceCallChatHistoryPersistingChatClient"/>,
/// ensuring that chat history is correctly persisted across multi-turn approval interactions.
/// </summary>
public class ChatClientAgent_ApprovalsTests
{
    #region Agent-As-Tool Approval Tests

    /// <summary>
    /// Verifies that direct invocation of an agent with an approval-required tool returns an approval request
    /// and does not execute the protected tool before approval.
    /// </summary>
    [Fact]
    public async Task RunAsync_ApprovalRequired_DirectAgentInvocation_DoesNotExecuteToolBeforeApprovalAsync()
    {
        // Arrange
        int protectedToolExecutionCount = 0;
        var protectedTool = new ApprovalRequiredAIFunction(AIFunctionFactory.Create(
            () =>
            {
                protectedToolExecutionCount++;
                return "protected result";
            },
            "ProtectedTool",
            "A protected tool"));

        var callIndex = new ChatClientAgentTestHelper.Ref<int>(0);
        var capturedInputs = new List<List<ChatMessage>>();
        var serviceExpectations = new List<ChatClientAgentTestHelper.ServiceCallExpectation>
        {
            new(new ChatResponse([new(ChatRole.Assistant,
                [new FunctionCallContent("childCall1", "ProtectedTool", new Dictionary<string, object?>())])])),
            new(new ChatResponse([new(ChatRole.Assistant, "Child final result.")])),
        };

        // Act
        var result1 = await ChatClientAgentTestHelper.RunAsync(
            inputMessages: [new(ChatRole.User, "Run the protected action.")],
            serviceCallExpectations: serviceExpectations,
            agentOptions: new() { ChatOptions = new() { Tools = [protectedTool] } },
            callIndex: callIndex,
            capturedInputs: capturedInputs);

        // Assert
        var approvalRequest = Assert.Single(result1.Response.Messages.SelectMany(m => m.Contents).OfType<ToolApprovalRequestContent>());
        Assert.Equal(0, protectedToolExecutionCount);

        // Act
        var result2 = await ChatClientAgentTestHelper.RunAsync(
            inputMessages: [new(ChatRole.User, [approvalRequest.CreateResponse(approved: true)])],
            serviceCallExpectations: serviceExpectations,
            existingSession: result1.Session,
            existingAgent: result1.Agent,
            existingMock: result1.MockService,
            callIndex: callIndex,
            capturedInputs: capturedInputs,
            expectedServiceCallCount: 2);

        // Assert
        Assert.Equal(1, protectedToolExecutionCount);
        Assert.Equal("Child final result.", result2.Response.Text);
    }

    /// <summary>
    /// Verifies that an approval-required tool inside a child agent still requires approval when the child
    /// agent is invoked as a tool by a parent agent.
    /// </summary>
    [Fact]
    public async Task RunAsync_ApprovalRequired_AgentAsTool_ApprovalResumesChildAgentAsync()
    {
        // Arrange
        int protectedToolExecutionCount = 0;
        var protectedTool = new ApprovalRequiredAIFunction(AIFunctionFactory.Create(
            () =>
            {
                protectedToolExecutionCount++;
                return "protected child result";
            },
            "ProtectedTool",
            "A protected child tool"));

        var childCallIndex = new ChatClientAgentTestHelper.Ref<int>(0);
        var childCapturedInputs = new List<List<ChatMessage>>();
        var childExpectations = new List<ChatClientAgentTestHelper.ServiceCallExpectation>
        {
            new(new ChatResponse([new(ChatRole.Assistant,
                [new FunctionCallContent("childCall1", "ProtectedTool", new Dictionary<string, object?>())])])),
            new(new ChatResponse([new(ChatRole.Assistant, "Child final result.")])),
        };

        var childMock = ChatClientAgentTestHelper.CreateSequentialMock(childExpectations, childCallIndex, childCapturedInputs);
        var childAgent = new ChatClientAgent(
            childMock.Object,
            options: new() { Name = "ChildAgent", Description = "Child approval agent", ChatOptions = new() { Tools = [protectedTool] } });

        AIFunction childAsTool = childAgent.AsAIFunction();

        var parentCallIndex = new ChatClientAgentTestHelper.Ref<int>(0);
        var parentCapturedInputs = new List<List<ChatMessage>>();
        var parentExpectations = new List<ChatClientAgentTestHelper.ServiceCallExpectation>
        {
            new(new ChatResponse([new(ChatRole.Assistant,
                [new FunctionCallContent("parentCall1", childAsTool.Name, new Dictionary<string, object?> { ["query"] = "Run child approval." })])])),
            new(
                new ChatResponse([new(ChatRole.Assistant, "Parent final result.")]),
                messages =>
                {
                    Assert.Contains(messages, m => m.Contents.OfType<FunctionCallContent>().Any(fcc => fcc.CallId == "parentCall1"));
                    Assert.Contains(messages, m => m.Contents.OfType<FunctionResultContent>().Any(frc => frc.CallId == "parentCall1" && frc.Result?.ToString() == "Child final result."));
                }),
        };

        // Act
        var result1 = await ChatClientAgentTestHelper.RunAsync(
            inputMessages: [new(ChatRole.User, "Ask the child agent to run the protected action.")],
            serviceCallExpectations: parentExpectations,
            agentOptions: new() { ChatOptions = new() { Tools = [childAsTool] } },
            callIndex: parentCallIndex,
            capturedInputs: parentCapturedInputs);

        // Assert
        var approvalRequest = Assert.Single(result1.Response.Messages.SelectMany(m => m.Contents).OfType<ToolApprovalRequestContent>());
        var parentToolCall = Assert.IsType<FunctionCallContent>(approvalRequest.ToolCall);
        Assert.Equal(childAsTool.Name, parentToolCall.Name);
        Assert.Equal("parentCall1", approvalRequest.RequestId);
        Assert.Equal(0, protectedToolExecutionCount);
        Assert.Equal(1, childCallIndex.Value);
        Assert.Equal(1, parentCallIndex.Value);

        // Act
        var result2 = await ChatClientAgentTestHelper.RunAsync(
            inputMessages: [new(ChatRole.User, [approvalRequest.CreateResponse(approved: true)])],
            serviceCallExpectations: parentExpectations,
            existingSession: result1.Session,
            existingAgent: result1.Agent,
            existingMock: result1.MockService,
            callIndex: parentCallIndex,
            capturedInputs: parentCapturedInputs,
            expectedServiceCallCount: 2);

        // Assert
        Assert.Equal(1, protectedToolExecutionCount);
        Assert.Equal(2, childCallIndex.Value);
        Assert.Equal("Parent final result.", result2.Response.Text);
    }

    /// <summary>
    /// Verifies that a child agent with a normal, non-approval tool continues to behave as a regular
    /// agent-as-tool invocation.
    /// </summary>
    [Fact]
    public async Task RunAsync_NonApprovalTool_AgentAsTool_CompletesWithoutApprovalAsync()
    {
        // Arrange
        int normalToolExecutionCount = 0;
        var normalTool = AIFunctionFactory.Create(
            () =>
            {
                normalToolExecutionCount++;
                return "normal child result";
            },
            "NormalTool",
            "A normal child tool");

        var childCallIndex = new ChatClientAgentTestHelper.Ref<int>(0);
        var childCapturedInputs = new List<List<ChatMessage>>();
        var childExpectations = new List<ChatClientAgentTestHelper.ServiceCallExpectation>
        {
            new(new ChatResponse([new(ChatRole.Assistant,
                [new FunctionCallContent("childCall1", "NormalTool", new Dictionary<string, object?>())])])),
            new(new ChatResponse([new(ChatRole.Assistant, "Child final result.")])),
        };

        var childMock = ChatClientAgentTestHelper.CreateSequentialMock(childExpectations, childCallIndex, childCapturedInputs);
        var childAgent = new ChatClientAgent(
            childMock.Object,
            options: new() { Name = "ChildAgent", Description = "Child normal agent", ChatOptions = new() { Tools = [normalTool] } });

        AIFunction childAsTool = childAgent.AsAIFunction();

        var parentCallIndex = new ChatClientAgentTestHelper.Ref<int>(0);
        var parentCapturedInputs = new List<List<ChatMessage>>();
        var parentExpectations = new List<ChatClientAgentTestHelper.ServiceCallExpectation>
        {
            new(new ChatResponse([new(ChatRole.Assistant,
                [new FunctionCallContent("parentCall1", childAsTool.Name, new Dictionary<string, object?> { ["query"] = "Run child normal tool." })])])),
            new(
                new ChatResponse([new(ChatRole.Assistant, "Parent final result.")]),
                messages => Assert.Contains(messages, m => m.Contents.OfType<FunctionResultContent>().Any(frc => frc.CallId == "parentCall1" && frc.Result?.ToString() == "Child final result."))),
        };

        // Act
        var result = await ChatClientAgentTestHelper.RunAsync(
            inputMessages: [new(ChatRole.User, "Ask the child agent to run the normal action.")],
            serviceCallExpectations: parentExpectations,
            agentOptions: new() { ChatOptions = new() { Tools = [childAsTool] } },
            callIndex: parentCallIndex,
            capturedInputs: parentCapturedInputs,
            expectedServiceCallCount: 2);

        // Assert
        Assert.Empty(result.Response.Messages.SelectMany(m => m.Contents).OfType<ToolApprovalRequestContent>());
        Assert.Equal(1, normalToolExecutionCount);
        Assert.Equal(2, childCallIndex.Value);
        Assert.Equal("Parent final result.", result.Response.Text);
    }

    /// <summary>
    /// Verifies that rejecting an approval request from a child agent-as-tool invocation clears the pending
    /// child invocation and does not execute the protected child tool.
    /// </summary>
    [Fact]
    public async Task RunAsync_ApprovalRejected_AgentAsTool_ClearsPendingChildInvocationAsync()
    {
        // Arrange
        int protectedToolExecutionCount = 0;
        var protectedTool = new ApprovalRequiredAIFunction(AIFunctionFactory.Create(
            () =>
            {
                protectedToolExecutionCount++;
                return "protected child result";
            },
            "ProtectedTool",
            "A protected child tool"));

        var childCallIndex = new ChatClientAgentTestHelper.Ref<int>(0);
        var childCapturedInputs = new List<List<ChatMessage>>();
        var childExpectations = new List<ChatClientAgentTestHelper.ServiceCallExpectation>
        {
            new(new ChatResponse([new(ChatRole.Assistant,
                [new FunctionCallContent("childCall1", "ProtectedTool", new Dictionary<string, object?>())])])),
        };

        var childMock = ChatClientAgentTestHelper.CreateSequentialMock(childExpectations, childCallIndex, childCapturedInputs);
        var childAgent = new ChatClientAgent(
            childMock.Object,
            options: new() { Name = "ChildAgent", Description = "Child rejected approval agent", ChatOptions = new() { Tools = [protectedTool] } });

        AIFunction childAsTool = childAgent.AsAIFunction();

        var parentCallIndex = new ChatClientAgentTestHelper.Ref<int>(0);
        var parentCapturedInputs = new List<List<ChatMessage>>();
        var parentExpectations = new List<ChatClientAgentTestHelper.ServiceCallExpectation>
        {
            new(new ChatResponse([new(ChatRole.Assistant,
                [new FunctionCallContent("parentCall1", childAsTool.Name, new Dictionary<string, object?> { ["query"] = "Run child approval." })])])),
            new(
                new ChatResponse([new(ChatRole.Assistant, "Parent handled rejection.")]),
                messages => Assert.Contains(messages, m => m.Contents.OfType<FunctionResultContent>().Any(
                    frc => frc.CallId == "parentCall1" &&
                        frc.Result?.ToString()?.Contains("rejected", StringComparison.OrdinalIgnoreCase) == true &&
                        frc.Result?.ToString()?.Contains("User declined", StringComparison.Ordinal) == true))),
        };

        var result1 = await ChatClientAgentTestHelper.RunAsync(
            inputMessages: [new(ChatRole.User, "Ask the child agent to run the protected action.")],
            serviceCallExpectations: parentExpectations,
            agentOptions: new() { ChatOptions = new() { Tools = [childAsTool] } },
            callIndex: parentCallIndex,
            capturedInputs: parentCapturedInputs);

        var approvalRequest = Assert.Single(result1.Response.Messages.SelectMany(m => m.Contents).OfType<ToolApprovalRequestContent>());

        // Act
        var result2 = await ChatClientAgentTestHelper.RunAsync(
            inputMessages: [new(ChatRole.User, [approvalRequest.CreateResponse(approved: false, reason: "User declined")])],
            serviceCallExpectations: parentExpectations,
            existingSession: result1.Session,
            existingAgent: result1.Agent,
            existingMock: result1.MockService,
            callIndex: parentCallIndex,
            capturedInputs: parentCapturedInputs,
            expectedServiceCallCount: 2);

        // Assert
        Assert.Equal(0, protectedToolExecutionCount);
        Assert.Equal(1, childCallIndex.Value);
        Assert.Equal(2, parentCallIndex.Value);
        Assert.Equal("Parent handled rejection.", result2.Response.Text);
    }

    /// <summary>
    /// Verifies that an approval-required tool inside a child agent still requires approval when the child
    /// agent is invoked as a tool by a streaming parent agent.
    /// </summary>
    [Fact]
    public async Task RunStreamingAsync_ApprovalRequired_AgentAsTool_ApprovalResumesChildAgentAsync()
    {
        // Arrange
        int protectedToolExecutionCount = 0;
        var protectedTool = new ApprovalRequiredAIFunction(AIFunctionFactory.Create(
            () =>
            {
                protectedToolExecutionCount++;
                return "protected child result";
            },
            "ProtectedTool",
            "A protected child tool"));

        var childCallIndex = new ChatClientAgentTestHelper.Ref<int>(0);
        var childCapturedInputs = new List<List<ChatMessage>>();
        var childExpectations = new List<ChatClientAgentTestHelper.ServiceCallExpectation>
        {
            new(new ChatResponse([new(ChatRole.Assistant,
                [new FunctionCallContent("childCall1", "ProtectedTool", new Dictionary<string, object?>())])])),
            new(new ChatResponse([new(ChatRole.Assistant, "Child final result.")])),
        };

        var childMock = ChatClientAgentTestHelper.CreateSequentialMock(childExpectations, childCallIndex, childCapturedInputs);
        var childAgent = new ChatClientAgent(
            childMock.Object,
            options: new() { Name = "ChildAgent", Description = "Child streaming approval agent", ChatOptions = new() { Tools = [protectedTool] } });

        AIFunction childAsTool = childAgent.AsAIFunction();

        var parentCallIndex = new ChatClientAgentTestHelper.Ref<int>(0);
        var parentCapturedInputs = new List<List<ChatMessage>>();
        var parentExpectations = new List<StreamingServiceCallExpectation>
        {
            new(
                [
                    new ChatResponseUpdate
                    {
                        Contents = [new FunctionCallContent("parentCall1", childAsTool.Name, new Dictionary<string, object?> { ["query"] = "Run child approval." })],
                        FinishReason = ChatFinishReason.ToolCalls,
                    },
                    new ChatResponseUpdate(ChatRole.Assistant, string.Empty),
                ]),
            new(
                [new ChatResponseUpdate(ChatRole.Assistant, "Parent final "), new ChatResponseUpdate(ChatRole.Assistant, "result.")],
                messages =>
                {
                    Assert.Contains(messages, m => m.Contents.OfType<FunctionCallContent>().Any(fcc => fcc.CallId == "parentCall1"));
                    Assert.Contains(messages, m => m.Contents.OfType<FunctionResultContent>().Any(frc => frc.CallId == "parentCall1" && frc.Result?.ToString() == "Child final result."));
                }),
        };

        Mock<IChatClient> parentMock = CreateSequentialStreamingMock(parentExpectations, parentCallIndex, parentCapturedInputs);
        var parentAgent = new ChatClientAgent(
            parentMock.Object,
            options: new() { ChatOptions = new() { Tools = [childAsTool] } },
            services: new ServiceCollection().BuildServiceProvider());

        var parentSession = (ChatClientAgentSession)await parentAgent.CreateSessionAsync();

        // Act
        var updates1 = await parentAgent.RunStreamingAsync(
            [new(ChatRole.User, "Ask the child agent to run the protected action.")],
            parentSession).ToListAsync();

        // Assert
        Assert.Equal(1, parentCallIndex.Value);
        Assert.Equal(1, childCallIndex.Value);
        Assert.NotEmpty(updates1);
        var approvalRequest = Assert.Single(updates1.SelectMany(u => u.Contents).OfType<ToolApprovalRequestContent>());
        var parentToolCall = Assert.IsType<FunctionCallContent>(approvalRequest.ToolCall);
        Assert.Equal(childAsTool.Name, parentToolCall.Name);
        Assert.Equal("parentCall1", approvalRequest.RequestId);
        Assert.Equal(0, protectedToolExecutionCount);
        Assert.Equal(1, childCallIndex.Value);
        Assert.Equal(1, parentCallIndex.Value);

        // Act
        var updates2 = await parentAgent.RunStreamingAsync(
            [new(ChatRole.User, [approvalRequest.CreateResponse(approved: true)])],
            parentSession).ToListAsync();

        // Assert
        Assert.Equal(1, protectedToolExecutionCount);
        Assert.Equal(2, childCallIndex.Value);
        Assert.Equal(2, parentCallIndex.Value);
        Assert.Equal("Parent final result.", string.Concat(updates2.Select(u => u.Text)));
    }

    /// <summary>
    /// Verifies that a streaming parent agent still completes normally when the child agent tool does not
    /// produce an approval request.
    /// </summary>
    [Fact]
    public async Task RunStreamingAsync_NonApprovalTool_AgentAsTool_CompletesWithoutApprovalAsync()
    {
        // Arrange
        int normalToolExecutionCount = 0;
        var normalTool = AIFunctionFactory.Create(
            () =>
            {
                normalToolExecutionCount++;
                return "normal child result";
            },
            "NormalTool",
            "A normal child tool");

        var childCallIndex = new ChatClientAgentTestHelper.Ref<int>(0);
        var childCapturedInputs = new List<List<ChatMessage>>();
        var childExpectations = new List<ChatClientAgentTestHelper.ServiceCallExpectation>
        {
            new(new ChatResponse([new(ChatRole.Assistant,
                [new FunctionCallContent("childCall1", "NormalTool", new Dictionary<string, object?>())])])),
            new(new ChatResponse([new(ChatRole.Assistant, "Child final result.")])),
        };

        var childMock = ChatClientAgentTestHelper.CreateSequentialMock(childExpectations, childCallIndex, childCapturedInputs);
        var childAgent = new ChatClientAgent(
            childMock.Object,
            options: new() { Name = "ChildAgent", Description = "Child streaming normal agent", ChatOptions = new() { Tools = [normalTool] } });

        AIFunction childAsTool = childAgent.AsAIFunction();

        var parentCallIndex = new ChatClientAgentTestHelper.Ref<int>(0);
        var parentCapturedInputs = new List<List<ChatMessage>>();
        var parentExpectations = new List<StreamingServiceCallExpectation>
        {
            new(
                [
                    new ChatResponseUpdate
                    {
                        Contents = [new FunctionCallContent("parentCall1", childAsTool.Name, new Dictionary<string, object?> { ["query"] = "Run child normal tool." })],
                        FinishReason = ChatFinishReason.ToolCalls,
                    },
                    new ChatResponseUpdate(ChatRole.Assistant, string.Empty),
                ]),
            new(
                [new ChatResponseUpdate(ChatRole.Assistant, "Parent final "), new ChatResponseUpdate(ChatRole.Assistant, "result.")],
                messages => Assert.Contains(messages, m => m.Contents.OfType<FunctionResultContent>().Any(frc => frc.CallId == "parentCall1" && frc.Result?.ToString() == "Child final result."))),
        };

        Mock<IChatClient> parentMock = CreateSequentialStreamingMock(parentExpectations, parentCallIndex, parentCapturedInputs);
        var parentAgent = new ChatClientAgent(
            parentMock.Object,
            options: new() { ChatOptions = new() { Tools = [childAsTool] } },
            services: new ServiceCollection().BuildServiceProvider());

        // Act
        var updates = await parentAgent.RunStreamingAsync(
            [new(ChatRole.User, "Ask the child agent to run the normal action.")]).ToListAsync();

        // Assert
        Assert.Empty(updates.SelectMany(u => u.Contents).OfType<ToolApprovalRequestContent>());
        Assert.Equal(1, normalToolExecutionCount);
        Assert.Equal(2, childCallIndex.Value);
        Assert.Equal(2, parentCallIndex.Value);
        Assert.Equal("Parent final result.", string.Concat(updates.Select(u => u.Text)));
    }

    #endregion

    #region Per-Service-Call Persistence Approval Tests

    /// <summary>
    /// Verifies that with per-service-call persistence and an approval-required tool,
    /// a two-turn approval flow persists the correct final history:
    /// Turn 1: user asks → model returns FCC → FICC converts to ToolApprovalRequestContent → returned to caller.
    /// Turn 2: caller sends ToolApprovalResponseContent → FICC processes approval, invokes function, calls model again.
    /// Final history: [user, assistant(FCC), tool(FRC), assistant(final)].
    /// </summary>
    [Fact]
    public async Task RunAsync_ApprovalRequired_PerServiceCallPersistence_PersistsCorrectHistoryAsync()
    {
        // Arrange
        var tool = AIFunctionFactory.Create(() => "Sunny, 22°C", "GetWeather", "Gets the weather");
        var approvalTool = new ApprovalRequiredAIFunction(tool);

        var callIndex = new ChatClientAgentTestHelper.Ref<int>(0);
        var capturedInputs = new List<List<ChatMessage>>();
        var serviceExpectations = new List<ChatClientAgentTestHelper.ServiceCallExpectation>
        {
            // Turn 1: model returns a function call (FICC will convert to approval request)
            new(new ChatResponse([new(ChatRole.Assistant,
                [new FunctionCallContent("call1", "GetWeather", new Dictionary<string, object?> { ["city"] = "Amsterdam" })])])),
            // Turn 2: after approval, FICC invokes the function and calls the model again
            new(new ChatResponse([new(ChatRole.Assistant, "The weather in Amsterdam is sunny and 22°C.")])),
        };

        // Act — Turn 1: initial request
        var result1 = await ChatClientAgentTestHelper.RunAsync(
            inputMessages: [new(ChatRole.User, "What's the weather?")],
            serviceCallExpectations: serviceExpectations,
            agentOptions: new()
            {
                ChatOptions = new() { Tools = [approvalTool] },
                RequirePerServiceCallChatHistoryPersistence = true,
            },
            callIndex: callIndex,
            capturedInputs: capturedInputs);

        // Verify Turn 1 returns exactly one approval request
        var approvalRequests = result1.Response.Messages
            .SelectMany(m => m.Contents)
            .OfType<ToolApprovalRequestContent>()
            .ToList();
        Assert.Single(approvalRequests);
        Assert.Equal(1, result1.TotalServiceCalls);

        // Verify service received user message on first call
        Assert.Single(capturedInputs);
        Assert.Contains(capturedInputs[0], m => m.Role == ChatRole.User && m.Text == "What's the weather?");

        // Act — Turn 2: send approval response
        var approvalResponseMessages = approvalRequests.ConvertAll(req =>
            new ChatMessage(ChatRole.User, [req.CreateResponse(approved: true)]));

        await ChatClientAgentTestHelper.RunAsync(
            inputMessages: approvalResponseMessages,
            serviceCallExpectations: serviceExpectations,
            existingSession: result1.Session,
            existingAgent: result1.Agent,
            existingMock: result1.MockService,
            callIndex: callIndex,
            capturedInputs: capturedInputs,
            expectedServiceCallCount: 2,
            expectedHistory:
            [
                new(ChatRole.User, TextContains: "What's the weather?"),
                new(ChatRole.Assistant, ContentTypes: [typeof(FunctionCallContent)]),
                new(ChatRole.Tool, ContentTypes: [typeof(FunctionResultContent)]),
                new(ChatRole.Assistant, TextContains: "sunny and 22°C"),
            ]);

        // Verify second service call received the full conversation (user + FCC + FRC)
        Assert.Equal(2, capturedInputs.Count);
        Assert.Contains(capturedInputs[1], m => m.Contents.OfType<FunctionCallContent>().Any());
        Assert.Contains(capturedInputs[1], m => m.Contents.OfType<FunctionResultContent>().Any());
    }

    #endregion

    #region End-of-Run Persistence Approval Tests

    /// <summary>
    /// Verifies that with end-of-run persistence and an approval-required tool,
    /// a two-turn approval flow persists the correct final history.
    /// </summary>
    [Fact]
    public async Task RunAsync_ApprovalRequired_EndOfRunPersistence_PersistsCorrectHistoryAsync()
    {
        // Arrange
        var tool = AIFunctionFactory.Create(() => "Sunny, 22°C", "GetWeather", "Gets the weather");
        var approvalTool = new ApprovalRequiredAIFunction(tool);

        var callIndex = new ChatClientAgentTestHelper.Ref<int>(0);
        var capturedInputs = new List<List<ChatMessage>>();
        var serviceExpectations = new List<ChatClientAgentTestHelper.ServiceCallExpectation>
        {
            new(new ChatResponse([new(ChatRole.Assistant,
                [new FunctionCallContent("call1", "GetWeather", new Dictionary<string, object?> { ["city"] = "Amsterdam" })])])),
            new(new ChatResponse([new(ChatRole.Assistant, "The weather in Amsterdam is sunny and 22°C.")])),
        };

        // Act — Turn 1
        var result1 = await ChatClientAgentTestHelper.RunAsync(
            inputMessages: [new(ChatRole.User, "What's the weather?")],
            serviceCallExpectations: serviceExpectations,
            agentOptions: new()
            {
                ChatOptions = new() { Tools = [approvalTool] },
            },
            callIndex: callIndex,
            capturedInputs: capturedInputs);

        var approvalRequests = result1.Response.Messages
            .SelectMany(m => m.Contents)
            .OfType<ToolApprovalRequestContent>()
            .ToList();
        Assert.Single(approvalRequests);

        // Act — Turn 2
        var approvalResponseMessages = approvalRequests.ConvertAll(req =>
            new ChatMessage(ChatRole.User, [req.CreateResponse(approved: true)]));

        var result2 = await ChatClientAgentTestHelper.RunAsync(
            inputMessages: approvalResponseMessages,
            serviceCallExpectations: serviceExpectations,
            existingSession: result1.Session,
            existingAgent: result1.Agent,
            existingMock: result1.MockService,
            callIndex: callIndex,
            capturedInputs: capturedInputs,
            expectedServiceCallCount: 2,
            expectedHistory:
            [
                // End-of-run persistence retains the approval request from Turn 1
                // and the approval response from Turn 2
                new(ChatRole.User, TextContains: "What's the weather?"),
                new(ChatRole.Assistant, ContentTypes: [typeof(ToolApprovalRequestContent)]),
                new(ChatRole.User, ContentTypes: [typeof(ToolApprovalResponseContent)]),
                new(ChatRole.Assistant, ContentTypes: [typeof(FunctionCallContent)]),
                new(ChatRole.Tool, ContentTypes: [typeof(FunctionResultContent)]),
                new(ChatRole.Assistant, TextContains: "sunny and 22°C"),
            ]);
    }

    #endregion

    #region Service-Stored History Approval Tests

    /// <summary>
    /// Verifies that with service-stored history (ConversationId returned) and an approval-required tool,
    /// the two-turn approval flow completes without errors and the session gets the ConversationId.
    /// </summary>
    [Fact]
    public async Task RunAsync_ApprovalRequired_ServiceStoredHistory_CompletesWithoutErrorAsync()
    {
        // Arrange
        const string ConversationId = "thread-456";
        var tool = AIFunctionFactory.Create(() => "Sunny, 22°C", "GetWeather", "Gets the weather");
        var approvalTool = new ApprovalRequiredAIFunction(tool);

        var callIndex = new ChatClientAgentTestHelper.Ref<int>(0);
        var capturedInputs = new List<List<ChatMessage>>();
        var serviceExpectations = new List<ChatClientAgentTestHelper.ServiceCallExpectation>
        {
            new(new ChatResponse([new(ChatRole.Assistant,
                [new FunctionCallContent("call1", "GetWeather", new Dictionary<string, object?> { ["city"] = "Amsterdam" })])])
            {
                ConversationId = ConversationId,
            }),
            new(new ChatResponse([new(ChatRole.Assistant, "The weather in Amsterdam is sunny and 22°C.")])
            {
                ConversationId = ConversationId,
            }),
        };

        // Act — Turn 1
        var result1 = await ChatClientAgentTestHelper.RunAsync(
            inputMessages: [new(ChatRole.User, "What's the weather?")],
            serviceCallExpectations: serviceExpectations,
            agentOptions: new()
            {
                ChatOptions = new() { Tools = [approvalTool] },
            },
            callIndex: callIndex,
            capturedInputs: capturedInputs);

        var approvalRequests = result1.Response.Messages
            .SelectMany(m => m.Contents)
            .OfType<ToolApprovalRequestContent>()
            .ToList();
        Assert.Single(approvalRequests);
        Assert.Equal(ConversationId, result1.Session.ConversationId);

        // Act — Turn 2
        var approvalResponseMessages = approvalRequests.ConvertAll(req =>
            new ChatMessage(ChatRole.User, [req.CreateResponse(approved: true)]));

        var result2 = await ChatClientAgentTestHelper.RunAsync(
            inputMessages: approvalResponseMessages,
            serviceCallExpectations: serviceExpectations,
            existingSession: result1.Session,
            existingAgent: result1.Agent,
            existingMock: result1.MockService,
            callIndex: callIndex,
            capturedInputs: capturedInputs,
            expectedServiceCallCount: 2);

        // Assert — session should retain the ConversationId, response should be correct
        Assert.Equal(ConversationId, result2.Session.ConversationId);
        Assert.Contains(result2.Response.Messages, m => m.Text == "The weather in Amsterdam is sunny and 22°C.");
    }

    #endregion

    #region Approval Rejected Tests

    /// <summary>
    /// Verifies that when an approval is rejected, the rejection result is persisted in the history
    /// and the model receives the rejection information.
    /// </summary>
    [Fact]
    public async Task RunAsync_ApprovalRejected_PersistsRejectionInHistoryAsync()
    {
        // Arrange
        var tool = AIFunctionFactory.Create(() => "Sunny, 22°C", "GetWeather", "Gets the weather");
        var approvalTool = new ApprovalRequiredAIFunction(tool);

        var callIndex = new ChatClientAgentTestHelper.Ref<int>(0);
        var capturedInputs = new List<List<ChatMessage>>();
        var serviceExpectations = new List<ChatClientAgentTestHelper.ServiceCallExpectation>
        {
            // Turn 1: model requests function call
            new(new ChatResponse([new(ChatRole.Assistant,
                [new FunctionCallContent("call1", "GetWeather", new Dictionary<string, object?> { ["city"] = "Amsterdam" })])])),
            // Turn 2: after rejection, model gets the rejection info and responds accordingly
            new(new ChatResponse([new(ChatRole.Assistant, "I'm sorry, I cannot check the weather without your approval.")])),
        };

        // Act — Turn 1
        var result1 = await ChatClientAgentTestHelper.RunAsync(
            inputMessages: [new(ChatRole.User, "What's the weather?")],
            serviceCallExpectations: serviceExpectations,
            agentOptions: new()
            {
                ChatOptions = new() { Tools = [approvalTool] },
                RequirePerServiceCallChatHistoryPersistence = true,
            },
            callIndex: callIndex,
            capturedInputs: capturedInputs);

        var approvalRequests = result1.Response.Messages
            .SelectMany(m => m.Contents)
            .OfType<ToolApprovalRequestContent>()
            .ToList();
        Assert.Single(approvalRequests);

        // Act — Turn 2: reject the approval
        var rejectionMessages = approvalRequests.ConvertAll(req =>
            new ChatMessage(ChatRole.User, [req.CreateResponse(approved: false, reason: "User declined")]));

        var result2 = await ChatClientAgentTestHelper.RunAsync(
            inputMessages: rejectionMessages,
            serviceCallExpectations: serviceExpectations,
            existingSession: result1.Session,
            existingAgent: result1.Agent,
            existingMock: result1.MockService,
            callIndex: callIndex,
            capturedInputs: capturedInputs,
            expectedServiceCallCount: 2);

        // Assert — history should contain the rejection result (FRC with rejection)
        var history = ChatClientAgentTestHelper.GetPersistedHistory(result2.Agent, result2.Session);
        Assert.True(
            history.Count >= 3,
            $"Expected at least 3 messages in history, got {history.Count}.\n{ChatClientAgentTestHelper.FormatMessages(history)}");
        Assert.Contains(history, m => m.Role == ChatRole.User && m.Text == "What's the weather?");
        Assert.Contains(history, m => m.Contents.OfType<FunctionResultContent>().Any(
            frc => frc.Result?.ToString()?.Contains("rejected") == true));
        Assert.Contains(history, m => m.Role == ChatRole.Assistant &&
            m.Text == "I'm sorry, I cannot check the weather without your approval.");

        // Verify the second service call received the rejection FRC
        Assert.Equal(2, capturedInputs.Count);
        Assert.Contains(capturedInputs[1], m => m.Contents.OfType<FunctionResultContent>().Any(
            frc => frc.Result?.ToString()?.Contains("rejected") == true));
    }

    #endregion

    private sealed record StreamingServiceCallExpectation(
        IReadOnlyList<ChatResponseUpdate> Updates,
        Action<List<ChatMessage>>? VerifyInput = null);

    private static Mock<IChatClient> CreateSequentialStreamingMock(
        List<StreamingServiceCallExpectation> expectations,
        ChatClientAgentTestHelper.Ref<int> callIndex,
        List<List<ChatMessage>> capturedInputs)
    {
        Mock<IChatClient> mock = new();
        mock.Setup(s => s.GetStreamingResponseAsync(
                It.IsAny<IEnumerable<ChatMessage>>(),
                It.IsAny<ChatOptions>(),
                It.IsAny<CancellationToken>()))
            .Returns<IEnumerable<ChatMessage>, ChatOptions?, CancellationToken>((msgs, _, _) =>
            {
                int idx = callIndex.Value++;
                var messageList = msgs.ToList();
                capturedInputs.Add(messageList);

                if (idx >= expectations.Count)
                {
                    throw new InvalidOperationException(
                        $"Mock received unexpected streaming service call #{idx + 1}. Only {expectations.Count} call(s) were expected.");
                }

                var expectation = expectations[idx];
                expectation.VerifyInput?.Invoke(messageList);
                return ToAsyncEnumerableAsync(expectation.Updates);
            });

        return mock;
    }

    private static async IAsyncEnumerable<ChatResponseUpdate> ToAsyncEnumerableAsync(IEnumerable<ChatResponseUpdate> updates)
    {
        foreach (ChatResponseUpdate update in updates)
        {
            await Task.Yield();
            yield return update;
        }
    }
}
