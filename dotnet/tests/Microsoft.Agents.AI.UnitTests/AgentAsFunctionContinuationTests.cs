// Copyright (c) Microsoft. All rights reserved.

using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Text.Json;
using Microsoft.Extensions.AI;

namespace Microsoft.Agents.AI.UnitTests;

public class AgentAsFunctionContinuationTests
{
    [Fact]
    public void SerializeAndDeserialize_WithPendingApproval_PreservesState()
    {
        // Arrange
        using JsonDocument document = JsonDocument.Parse("""{"sessionId":"child-session"}""");
        var approvalRequest = new ToolApprovalRequestContent(
            "approval-1",
            new FunctionCallContent("child-call-1", "DeleteFile"));
        Dictionary<string, AgentAsFunctionContinuation> continuations = new()
        {
            ["continuation-1"] = new()
            {
                ParentCallName = "ChildAgent",
                ParentCallId = "parent-call-1",
                SerializedSession = document.RootElement.Clone(),
                PendingToolApprovalRequestDict =
                    new ConcurrentDictionary<string, ToolApprovalRequestContent>(
                        new Dictionary<string, ToolApprovalRequestContent>
                        {
                            [approvalRequest.RequestId] = approvalRequest,
                        }),
            },
        };

        // Act
        JsonElement serialized = JsonSerializer.SerializeToElement(
            continuations,
            AgentJsonUtilities.DefaultOptions);
        var deserialized = JsonSerializer.Deserialize<Dictionary<string, AgentAsFunctionContinuation>>(
            serialized,
            AgentJsonUtilities.DefaultOptions);

        // Assert
        Assert.NotNull(deserialized);
        var continuation = Assert.Single(deserialized).Value;
        Assert.Equal("ChildAgent", continuation.ParentCallName);
        Assert.Equal("parent-call-1", continuation.ParentCallId);
        Assert.Equal("child-session", continuation.SerializedSession.GetProperty("sessionId").GetString());

        var pendingRequest = Assert.Single(continuation.PendingToolApprovalRequestDict).Value;
        Assert.Equal("approval-1", pendingRequest.RequestId);
        var toolCall = Assert.IsType<FunctionCallContent>(pendingRequest.ToolCall);
        Assert.Equal("child-call-1", toolCall.CallId);
        Assert.Equal("DeleteFile", toolCall.Name);
    }
}
