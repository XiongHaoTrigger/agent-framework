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
/// Unit tests for the <see cref="AIAgentExtensions.AsAIFunction"/> method with approval support.
/// Verifies that tool approval requests from the inner agent are properly propagated when used as a parent agent tool.
/// </summary>
public class AsAIFunction_ApprovalTests
{
    [Fact]
    public void AsAIFunction_WithNullAgent_ThrowsArgumentNullException()
    {
        var exception = Assert.Throws<ArgumentNullException>(() =>
            AIAgentExtensions.AsAIFunction(null!));

        Assert.Equal("agent", exception.ParamName);
    }

    [Fact]
    public async Task AsAIFunction_WithoutApprovalRequests_ReturnsTextAsync()
    {
        const string ExpectedText = "Normal response text";
        var response = new AgentResponse(new ChatMessage(ChatRole.Assistant, ExpectedText));
        var testAgent = new ApprovalTestAgent(response);

        var aiFunction = testAgent.AsAIFunction();

        var result = await aiFunction.InvokeAsync(new AIFunctionArguments { ["query"] = "Test query" });

        Assert.NotNull(result);
        Assert.Equal(ExpectedText, result.ToString());
    }

    [Fact]
    public async Task AsAIFunction_WithApprovalRequestsNoHandler_ReturnsEmptyStringAsync()
    {
        var fcc = new FunctionCallContent("call1", "TestTool");
        var tarc = new ToolApprovalRequestContent("call1", fcc);
        var response = new AgentResponse(new ChatMessage(ChatRole.Assistant, [tarc]));
        var testAgent = new ApprovalTestAgent(response);

        var aiFunction = testAgent.AsAIFunction();

        var result = await aiFunction.InvokeAsync(new AIFunctionArguments { ["query"] = "Test query" });

        Assert.NotNull(result);
        Assert.Equal(string.Empty, result.ToString());
    }

    [Fact]
    public async Task AsAIFunction_WithAdditionalProperties_PropagatesToChildAgentAsync()
    {
        const string ExpectedText = "Response with additional properties";
        var response = new AgentResponse(new ChatMessage(ChatRole.Assistant, ExpectedText));
        var testAgent = new ApprovalTestAgent(response);
        var aiFunction = testAgent.AsAIFunction();

        var context = new FunctionInvocationContext()
        {
            Options = new()
            {
                AdditionalProperties = new AdditionalPropertiesDictionary
                {
                    { "customProperty1", "value1" },
                    { "customProperty2", 42 }
                }
            }
        };
        SetFunctionInvokingChatClientCurrentContext(context);

        var result = await aiFunction.InvokeAsync(new AIFunctionArguments { ["query"] = "Test query" });

        Assert.NotNull(result);
        Assert.Equal(ExpectedText, result.ToString());
        Assert.NotNull(testAgent.ReceivedAgentRunOptions);
        Assert.NotNull(testAgent.ReceivedAgentRunOptions!.AdditionalProperties);
        Assert.Equal("value1", testAgent.ReceivedAgentRunOptions!.AdditionalProperties["customProperty1"]);
        Assert.Equal(42, testAgent.ReceivedAgentRunOptions!.AdditionalProperties["customProperty2"]);
    }

    [Fact]
    public async Task AsAIFunction_AsParentTool_SurfacesChildApprovalRequestAsync()
    {
        // Arrange
        var childToolCall = new FunctionCallContent("childCall1", "DangerousChildTool");
        var childApprovalRequest = new ToolApprovalRequestContent("childRequest1", childToolCall);
        var childAgent = new ApprovalTestAgent(new AgentResponse(new ChatMessage(ChatRole.Assistant, [childApprovalRequest])));
        var childAsTool = childAgent.AsAIFunction();

        var parentToolCall = new FunctionCallContent(
            "parentCall1",
            childAsTool.Name,
            new Dictionary<string, object?> { ["query"] = "Run the child task" });

        var callIndex = new ChatClientAgentTestHelper.Ref<int>(0);
        var capturedInputs = new List<List<ChatMessage>>();

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
            callIndex: callIndex,
            capturedInputs: capturedInputs,
            expectedServiceCallCount: 1);

        // Assert
        ToolApprovalRequestContent surfacedApproval = result.Response.Messages
            .SelectMany(m => m.Contents)
            .OfType<ToolApprovalRequestContent>()
            .Single();

        Assert.Equal("childRequest1", surfacedApproval.RequestId);
        Assert.Same(childToolCall, surfacedApproval.ToolCall);
        AssertParentCallPrecedesApproval(result.Response, parentToolCall, surfacedApproval);
        Assert.Equal(1, childAgent.RunAsyncCallCount);
        Assert.DoesNotContain(result.Response.Messages.SelectMany(m => m.Contents), c =>
            c is FunctionResultContent frc &&
            frc.Result?.ToString()?.StartsWith(AgentAsFunctionApprovalDelegatingChatClient.PendingApprovalMarkerPrefix, StringComparison.Ordinal) == true);
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

        var parentToolCall = new FunctionCallContent(
            "parentCall1",
            childAsTool.Name,
            new Dictionary<string, object?> { ["query"] = "Run the child task" });

        var callIndex = new ChatClientAgentTestHelper.Ref<int>(0);
        var capturedInputs = new List<List<ChatMessage>>();
        var serviceCallExpectations = new List<ChatClientAgentTestHelper.ServiceCallExpectation>
        {
            new(new ChatResponse([new(ChatRole.Assistant, [parentToolCall])])),
            new(new ChatResponse([new(ChatRole.Assistant, "Parent observed child completion.")]),
                messages =>
                {
                    FunctionResultContent functionResult = messages
                        .SelectMany(m => m.Contents)
                        .OfType<FunctionResultContent>()
                        .Single();

                    Assert.Equal("parentCall1", functionResult.CallId);
                    Assert.Equal(FinalChildText, functionResult.Result);
                }),
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

        ToolApprovalRequestContent surfacedApproval = firstRun.Response.Messages
            .SelectMany(m => m.Contents)
            .OfType<ToolApprovalRequestContent>()
            .Single();

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
    public async Task AsAIFunction_AsParentTool_WithSequentialApprovalResponses_SurfacesNextApprovalBeforeParentContinuesAsync()
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

        var parentToolCall = new FunctionCallContent(
            "parentCall1",
            childAsTool.Name,
            new Dictionary<string, object?> { ["query"] = "Run the child task" });

        var callIndex = new ChatClientAgentTestHelper.Ref<int>(0);
        var capturedInputs = new List<List<ChatMessage>>();
        var serviceCallExpectations = new List<ChatClientAgentTestHelper.ServiceCallExpectation>
        {
            new(new ChatResponse([new(ChatRole.Assistant, [parentToolCall])])),
            new(new ChatResponse([new(ChatRole.Assistant, "Parent observed child completion.")]),
                messages =>
                {
                    FunctionResultContent functionResult = messages
                        .SelectMany(m => m.Contents)
                        .OfType<FunctionResultContent>()
                        .Single();

                    Assert.Equal("parentCall1", functionResult.CallId);
                    Assert.Equal(FinalChildText, functionResult.Result);
                }),
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

        ToolApprovalRequestContent firstSurfacedApproval = firstRun.Response.Messages
            .SelectMany(m => m.Contents)
            .OfType<ToolApprovalRequestContent>()
            .Single();

        Assert.Equal("childRequest1", firstSurfacedApproval.RequestId);
        Assert.Same(firstChildToolCall, firstSurfacedApproval.ToolCall);

        // Act 1: Approving the first child approval should surface the second child approval
        // without letting the parent model continue yet.
        var secondRun = await ChatClientAgentTestHelper.RunAsync(
            inputMessages: [new(ChatRole.User, [firstSurfacedApproval.CreateResponse(approved: true)])],
            serviceCallExpectations: serviceCallExpectations,
            existingSession: firstRun.Session,
            existingAgent: firstRun.Agent,
            existingMock: firstRun.MockService,
            callIndex: callIndex,
            capturedInputs: capturedInputs,
            expectedServiceCallCount: 1);

        ToolApprovalRequestContent secondSurfacedApproval = secondRun.Response.Messages
            .SelectMany(m => m.Contents)
            .OfType<ToolApprovalRequestContent>()
            .Single();

        Assert.Equal("childRequest2", secondSurfacedApproval.RequestId);
        Assert.Same(secondChildToolCall, secondSurfacedApproval.ToolCall);
        AssertParentCallPrecedesApproval(secondRun.Response, parentToolCall, secondSurfacedApproval);

        // Act 2: Only after approving the second child approval can the child complete and
        // return its final text as the parent tool result.
        var thirdRun = await ChatClientAgentTestHelper.RunAsync(
            inputMessages: [new(ChatRole.User, [secondSurfacedApproval.CreateResponse(approved: true)])],
            serviceCallExpectations: serviceCallExpectations,
            existingSession: firstRun.Session,
            existingAgent: firstRun.Agent,
            existingMock: firstRun.MockService,
            callIndex: callIndex,
            capturedInputs: capturedInputs,
            expectedServiceCallCount: 2);

        // Assert
        Assert.Equal(3, childAgent.RunAsyncCallCount);
        Assert.Equal("Parent observed child completion.", thirdRun.Response.Text);
    }

    [Fact]
    public async Task AsAIFunction_ThreeLevelNestedAgents_WithApprovalResponse_ResumesFullChainAsync()
    {
        // Arrange
        const string FinalGrandchildText = "Grandchild action completed.";
        const string FinalChildText = "Child observed grandchild completion.";
        const string FinalParentText = "Parent observed child completion.";

        var grandchildToolCall = new FunctionCallContent("grandchildCall1", "DangerousGrandchildTool");
        var grandchildApprovalRequest = new ToolApprovalRequestContent("grandchildRequest1", grandchildToolCall);
        var grandchildAgent = new ApprovalTestAgent(
            new AgentResponse(new ChatMessage(ChatRole.Assistant, [grandchildApprovalRequest])),
            new AgentResponse(new ChatMessage(ChatRole.Assistant, FinalGrandchildText)));
        AgentSession grandchildSession = await grandchildAgent.CreateSessionAsync();
        var grandchildAsTool = grandchildAgent.AsAIFunction(session: grandchildSession);

        var childToGrandchildCall = new FunctionCallContent(
            "childToGrandchildCall1",
            grandchildAsTool.Name,
            new Dictionary<string, object?> { ["query"] = "Run the grandchild task" });

        var childCallIndex = new ChatClientAgentTestHelper.Ref<int>(0);
        var childCapturedInputs = new List<List<ChatMessage>>();
        var childServiceExpectations = new List<ChatClientAgentTestHelper.ServiceCallExpectation>
        {
            new(new ChatResponse([new(ChatRole.Assistant, [childToGrandchildCall])])),
            new(new ChatResponse([new(ChatRole.Assistant, FinalChildText)]),
                messages =>
                {
                    FunctionResultContent functionResult = messages
                        .SelectMany(m => m.Contents)
                        .OfType<FunctionResultContent>()
                        .Single();

                    Assert.Equal("childToGrandchildCall1", functionResult.CallId);
                    Assert.Equal(FinalGrandchildText, functionResult.Result);
                }),
        };
        var childMock = ChatClientAgentTestHelper.CreateSequentialMock(childServiceExpectations, childCallIndex, childCapturedInputs);
        var childAgent = new ChatClientAgent(
            childMock.Object,
            options: new ChatClientAgentOptions
            {
                ChatOptions = new() { Tools = [grandchildAsTool] },
            },
            services: new ServiceCollection().BuildServiceProvider());
        var childSession = (ChatClientAgentSession)await childAgent.CreateSessionAsync();
        var childAsTool = childAgent.AsAIFunction(session: childSession);

        var parentToChildCall = new FunctionCallContent(
            "parentToChildCall1",
            childAsTool.Name,
            new Dictionary<string, object?> { ["query"] = "Run the child task" });

        var parentCallIndex = new ChatClientAgentTestHelper.Ref<int>(0);
        var parentCapturedInputs = new List<List<ChatMessage>>();
        var parentServiceExpectations = new List<ChatClientAgentTestHelper.ServiceCallExpectation>
        {
            new(new ChatResponse([new(ChatRole.Assistant, [parentToChildCall])])),
            new(new ChatResponse([new(ChatRole.Assistant, FinalParentText)]),
                messages =>
                {
                    FunctionResultContent functionResult = messages
                        .SelectMany(m => m.Contents)
                        .OfType<FunctionResultContent>()
                        .Single();

                    Assert.Equal("parentToChildCall1", functionResult.CallId);
                    Assert.Equal(FinalChildText, functionResult.Result);
                }),
        };

        var firstRun = await ChatClientAgentTestHelper.RunAsync(
            inputMessages: [new(ChatRole.User, "Delegate through child and grandchild agents")],
            serviceCallExpectations: parentServiceExpectations,
            agentOptions: new()
            {
                ChatOptions = new() { Tools = [childAsTool] },
            },
            callIndex: parentCallIndex,
            capturedInputs: parentCapturedInputs);

        ToolApprovalRequestContent surfacedApproval = firstRun.Response.Messages
            .SelectMany(m => m.Contents)
            .OfType<ToolApprovalRequestContent>()
            .Single();

        Assert.Equal("grandchildRequest1", surfacedApproval.RequestId);
        Assert.Same(grandchildToolCall, surfacedApproval.ToolCall);
        AssertParentCallPrecedesApproval(firstRun.Response, parentToChildCall, surfacedApproval);

        // Act
        var secondRun = await ChatClientAgentTestHelper.RunAsync(
            inputMessages: [new(ChatRole.User, [surfacedApproval.CreateResponse(approved: true)])],
            serviceCallExpectations: parentServiceExpectations,
            existingSession: firstRun.Session,
            existingAgent: firstRun.Agent,
            existingMock: firstRun.MockService,
            callIndex: parentCallIndex,
            capturedInputs: parentCapturedInputs,
            expectedServiceCallCount: 2);

        // Assert
        Assert.Equal(2, grandchildAgent.RunAsyncCallCount);
        Assert.Equal(2, childCallIndex.Value);
        Assert.Equal(FinalParentText, secondRun.Response.Text);
    }

    [Fact]
    public async Task AsAIFunction_AsParentTool_TextApproveDoesNotResumeChildAgentAsync()
    {
        // Arrange
        var childToolCall = new FunctionCallContent("childCall1", "DangerousChildTool");
        var childApprovalRequest = new ToolApprovalRequestContent("childRequest1", childToolCall);
        var childAgent = new ApprovalTestAgent(new AgentResponse(new ChatMessage(ChatRole.Assistant, [childApprovalRequest])));
        var childAsTool = childAgent.AsAIFunction();

        var parentToolCall = new FunctionCallContent(
            "parentCall1",
            childAsTool.Name,
            new Dictionary<string, object?> { ["query"] = "Run the child task" });

        var callIndex = new ChatClientAgentTestHelper.Ref<int>(0);
        var capturedInputs = new List<List<ChatMessage>>();
        var serviceCallExpectations = new List<ChatClientAgentTestHelper.ServiceCallExpectation>
        {
            new(new ChatResponse([new(ChatRole.Assistant, [parentToolCall])])),
            new(new ChatResponse([new(ChatRole.Assistant, "Still waiting for structured approval.")]),
                messages =>
                {
                    Assert.Contains(messages, m => m.Role == ChatRole.User && m.Text == "approve");
                    Assert.DoesNotContain(messages.SelectMany(m => m.Contents), c => c is ToolApprovalResponseContent);
                    Assert.DoesNotContain(messages.SelectMany(m => m.Contents), c => c is FunctionResultContent);
                }),
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

        Assert.Single(firstRun.Response.Messages.SelectMany(m => m.Contents).OfType<ToolApprovalRequestContent>());

        // Act
        var secondRun = await ChatClientAgentTestHelper.RunAsync(
            inputMessages: [new(ChatRole.User, "approve")],
            serviceCallExpectations: serviceCallExpectations,
            existingSession: firstRun.Session,
            existingAgent: firstRun.Agent,
            existingMock: firstRun.MockService,
            callIndex: callIndex,
            capturedInputs: capturedInputs,
            expectedServiceCallCount: 2);

        // Assert
        Assert.Equal(1, childAgent.RunAsyncCallCount);
        Assert.Equal("Still waiting for structured approval.", secondRun.Response.Text);
    }

    [Fact]
    public async Task AsAIFunction_AsParentToolStreaming_SurfacesChildApprovalRequestAsync()
    {
        // Arrange
        var childToolCall = new FunctionCallContent("childCall1", "DangerousChildTool");
        var childApprovalRequest = new ToolApprovalRequestContent("childRequest1", childToolCall);
        var childAgent = new ApprovalTestAgent(new AgentResponse(new ChatMessage(ChatRole.Assistant, [childApprovalRequest])));
        var childAsTool = childAgent.AsAIFunction();

        var parentToolCall = new FunctionCallContent(
            "parentCall1",
            childAsTool.Name,
            new Dictionary<string, object?> { ["query"] = "Run the child task" });

        var callIndex = new ChatClientAgentTestHelper.Ref<int>(0);

        // Act
        var result = await RunStreamingAsync(
            inputMessages: [new(ChatRole.User, "Delegate to the child agent")],
            serviceCallExpectations:
            [
                new([new ChatResponseUpdate(ChatRole.Assistant, [parentToolCall])])
            ],
            agentOptions: new()
            {
                ChatOptions = new() { Tools = [childAsTool] },
            },
            callIndex: callIndex,
            expectedServiceCallCount: 1);

        // Assert
        ToolApprovalRequestContent surfacedApproval = result.Response.Messages
            .SelectMany(m => m.Contents)
            .OfType<ToolApprovalRequestContent>()
            .Single();

        Assert.Equal("childRequest1", surfacedApproval.RequestId);
        Assert.Same(childToolCall, surfacedApproval.ToolCall);
        AssertParentCallPrecedesApproval(result.Response, parentToolCall, surfacedApproval);
        Assert.Equal(1, childAgent.RunAsyncCallCount);
        Assert.DoesNotContain(result.Response.Messages.SelectMany(m => m.Contents), c =>
            c is FunctionResultContent frc &&
            frc.Result?.ToString()?.StartsWith(AgentAsFunctionApprovalDelegatingChatClient.PendingApprovalMarkerPrefix, StringComparison.Ordinal) == true);
    }

    [Fact]
    public async Task AsAIFunction_AsParentToolStreaming_WithApprovalResponse_ResumesChildAgentAsync()
    {
        // Arrange
        const string FinalChildText = "Child action completed.";
        var childToolCall = new FunctionCallContent("childCall1", "DangerousChildTool");
        var childApprovalRequest = new ToolApprovalRequestContent("childRequest1", childToolCall);
        var childAgent = new ApprovalTestAgent(
            new AgentResponse(new ChatMessage(ChatRole.Assistant, [childApprovalRequest])),
            new AgentResponse(new ChatMessage(ChatRole.Assistant, FinalChildText)));
        var childAsTool = childAgent.AsAIFunction();

        var parentToolCall = new FunctionCallContent(
            "parentCall1",
            childAsTool.Name,
            new Dictionary<string, object?> { ["query"] = "Run the child task" });

        var callIndex = new ChatClientAgentTestHelper.Ref<int>(0);
        var serviceCallExpectations = new List<StreamingServiceCallExpectation>
        {
            new([new ChatResponseUpdate(ChatRole.Assistant, [parentToolCall])]),
            new(
                [
                    new ChatResponseUpdate(ChatRole.Assistant, "Parent observed "),
                    new ChatResponseUpdate(ChatRole.Assistant, "child completion.")
                ],
                messages =>
                {
                    FunctionResultContent functionResult = messages
                        .SelectMany(m => m.Contents)
                        .OfType<FunctionResultContent>()
                        .Single();

                    Assert.Equal("parentCall1", functionResult.CallId);
                    Assert.Equal(FinalChildText, functionResult.Result);
                }),
        };

        var firstRun = await RunStreamingAsync(
            inputMessages: [new(ChatRole.User, "Delegate to the child agent")],
            serviceCallExpectations: serviceCallExpectations,
            agentOptions: new()
            {
                ChatOptions = new() { Tools = [childAsTool] },
            },
            callIndex: callIndex);

        ToolApprovalRequestContent surfacedApproval = firstRun.Response.Messages
            .SelectMany(m => m.Contents)
            .OfType<ToolApprovalRequestContent>()
            .Single();

        // Act
        var secondRun = await RunStreamingAsync(
            inputMessages: [new(ChatRole.User, [surfacedApproval.CreateResponse(approved: true)])],
            serviceCallExpectations: serviceCallExpectations,
            existingSession: firstRun.Session,
            existingAgent: firstRun.Agent,
            existingMock: firstRun.MockService,
            callIndex: callIndex,
            expectedServiceCallCount: 2);

        // Assert
        Assert.Equal(2, childAgent.RunAsyncCallCount);
        Assert.Equal("Parent observed child completion.", secondRun.Response.Text);
    }

    [Fact]
    public async Task AsAIFunction_AsParentToolStreaming_TextApproveDoesNotResumeChildAgentAsync()
    {
        // Arrange
        var childToolCall = new FunctionCallContent("childCall1", "DangerousChildTool");
        var childApprovalRequest = new ToolApprovalRequestContent("childRequest1", childToolCall);
        var childAgent = new ApprovalTestAgent(new AgentResponse(new ChatMessage(ChatRole.Assistant, [childApprovalRequest])));
        var childAsTool = childAgent.AsAIFunction();

        var parentToolCall = new FunctionCallContent(
            "parentCall1",
            childAsTool.Name,
            new Dictionary<string, object?> { ["query"] = "Run the child task" });

        var callIndex = new ChatClientAgentTestHelper.Ref<int>(0);
        var serviceCallExpectations = new List<StreamingServiceCallExpectation>
        {
            new([new ChatResponseUpdate(ChatRole.Assistant, [parentToolCall])]),
            new([new ChatResponseUpdate(ChatRole.Assistant, "Still waiting for structured approval.")],
                messages =>
                {
                    Assert.Contains(messages, m => m.Role == ChatRole.User && m.Text == "approve");
                    Assert.DoesNotContain(messages.SelectMany(m => m.Contents), c => c is ToolApprovalResponseContent);
                    Assert.DoesNotContain(messages.SelectMany(m => m.Contents), c => c is FunctionResultContent);
                }),
        };

        var firstRun = await RunStreamingAsync(
            inputMessages: [new(ChatRole.User, "Delegate to the child agent")],
            serviceCallExpectations: serviceCallExpectations,
            agentOptions: new()
            {
                ChatOptions = new() { Tools = [childAsTool] },
            },
            callIndex: callIndex);

        Assert.Single(firstRun.Response.Messages.SelectMany(m => m.Contents).OfType<ToolApprovalRequestContent>());

        // Act
        var secondRun = await RunStreamingAsync(
            inputMessages: [new(ChatRole.User, "approve")],
            serviceCallExpectations: serviceCallExpectations,
            existingSession: firstRun.Session,
            existingAgent: firstRun.Agent,
            existingMock: firstRun.MockService,
            callIndex: callIndex,
            expectedServiceCallCount: 2);

        // Assert
        Assert.Equal(1, childAgent.RunAsyncCallCount);
        Assert.Equal("Still waiting for structured approval.", secondRun.Response.Text);
    }

    private static void SetFunctionInvokingChatClientCurrentContext(FunctionInvocationContext? context)
    {
        var currentContextField = typeof(FunctionInvokingChatClient).GetField(
            "_currentContext",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);

        if (currentContextField?.GetValue(null) is AsyncLocal<FunctionInvocationContext?> asyncLocal)
        {
            asyncLocal.Value = context;
        }
    }

    private static void AssertParentCallPrecedesApproval(
        AgentResponse response,
        FunctionCallContent parentToolCall,
        ToolApprovalRequestContent surfacedApproval)
    {
        var contents = response.Messages.SelectMany(m => m.Contents).ToList();
        int parentCallIndex = contents.FindIndex(c => ReferenceEquals(parentToolCall, c));
        int approvalIndex = contents.FindIndex(c => ReferenceEquals(surfacedApproval, c));

        Assert.True(parentCallIndex >= 0, "The surfaced transcript should retain the parent agent tool call.");
        Assert.True(approvalIndex >= 0, "The surfaced transcript should contain the child approval request.");
        Assert.True(parentCallIndex < approvalIndex, "The child approval request should follow the parent agent tool call.");
    }

    private static Mock<IChatClient> CreateSequentialStreamingMock(
        List<StreamingServiceCallExpectation> expectations,
        ChatClientAgentTestHelper.Ref<int> callIndex)
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

    private static async Task<ChatClientAgentTestHelper.RunResult> RunStreamingAsync(
        List<ChatMessage> inputMessages,
        List<StreamingServiceCallExpectation> serviceCallExpectations,
        ChatClientAgentOptions? agentOptions = null,
        ChatClientAgentSession? existingSession = null,
        ChatClientAgent? existingAgent = null,
        Mock<IChatClient>? existingMock = null,
        ChatClientAgentTestHelper.Ref<int>? callIndex = null,
        int? expectedServiceCallCount = null)
    {
        callIndex ??= new ChatClientAgentTestHelper.Ref<int>(0);
        var mock = existingMock ?? CreateSequentialStreamingMock(serviceCallExpectations, callIndex);
        agentOptions ??= new ChatClientAgentOptions();

        var agent = existingAgent ?? new ChatClientAgent(
            mock.Object,
            options: agentOptions,
            services: new ServiceCollection().BuildServiceProvider());

        var session = existingSession ?? (await agent.CreateSessionAsync() as ChatClientAgentSession)!;
        AgentResponse response = await agent.RunStreamingAsync(inputMessages, session).ToAgentResponseAsync();

        if (expectedServiceCallCount.HasValue)
        {
            Assert.Equal(expectedServiceCallCount.Value, callIndex.Value);
        }

        return new ChatClientAgentTestHelper.RunResult(response, session, agent, mock, callIndex.Value, []);
    }

    private static async IAsyncEnumerable<ChatResponseUpdate> ToAsyncEnumerableAsync(IEnumerable<ChatResponseUpdate> updates)
    {
        foreach (var update in updates)
        {
            yield return update;
        }

        await Task.CompletedTask;
    }

    private sealed record StreamingServiceCallExpectation(
        ChatResponseUpdate[] Updates,
        Action<List<ChatMessage>>? VerifyInput = null);

    private sealed class ApprovalTestAgent : AIAgent
    {
        private readonly AgentResponse[] _responses;
        private int _callIndex;

        public ApprovalTestAgent(params AgentResponse[] responses)
        {
            _responses = responses;
        }

        public override string? Name => "TestApprovalAgent";
        public override string? Description => "Test agent for approval tests";

        public List<ChatMessage> ReceivedMessages { get; } = [];
        public AgentRunOptions? ReceivedAgentRunOptions { get; private set; }
        public int RunAsyncCallCount => _callIndex;

        protected override ValueTask<AgentSession> CreateSessionCoreAsync(CancellationToken cancellationToken = default)
            => new(new TestAgentSession());

        protected override ValueTask<JsonElement> SerializeSessionCoreAsync(AgentSession session, JsonSerializerOptions? jsonSerializerOptions = null, CancellationToken cancellationToken = default)
            => new(JsonSerializer.SerializeToElement(new SerializedTestAgentSession(session.StateBag), jsonSerializerOptions));

        protected override ValueTask<AgentSession> DeserializeSessionCoreAsync(JsonElement serializedState, JsonSerializerOptions? jsonSerializerOptions = null, CancellationToken cancellationToken = default)
        {
            var session = serializedState.Deserialize<SerializedTestAgentSession>(jsonSerializerOptions);
            return new(new TestAgentSession(session?.StateBag ?? new()));
        }

        protected override Task<AgentResponse> RunCoreAsync(
            IEnumerable<ChatMessage> messages,
            AgentSession? session = null,
            AgentRunOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            int index = _callIndex++;
            this.ReceivedMessages.AddRange(messages);
            this.ReceivedAgentRunOptions = options;

            if (index >= _responses.Length)
            {
                throw new InvalidOperationException(
                    $"Test agent received unexpected call #{index + 1}. Only {_responses.Length} response(s) were configured.");
            }

            return Task.FromResult(_responses[index]);
        }

        protected override async IAsyncEnumerable<AgentResponseUpdate> RunCoreStreamingAsync(
            IEnumerable<ChatMessage> messages,
            AgentSession? session = null,
            AgentRunOptions? options = null,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var response = await this.RunAsync(messages, session, options, cancellationToken);
            foreach (var update in response.ToAgentResponseUpdates())
            {
                yield return update;
            }
        }
    }

    private sealed class TestAgentSession : AgentSession
    {
        public TestAgentSession()
        {
        }

        public TestAgentSession(AgentSessionStateBag stateBag)
            : base(stateBag)
        {
        }
    }

    private sealed record SerializedTestAgentSession(AgentSessionStateBag StateBag);
}
