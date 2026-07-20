// Copyright (c) Microsoft. All rights reserved.

using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.AI;
using Moq;

namespace Microsoft.Agents.AI.UnitTests;

public class ChatClientAgent_AgentAsFunctionApprovalsTests(ITestOutputHelper output)
{
    private readonly ITestOutputHelper _output = output;

    [Fact]
    public async Task RunAsync_AgentAsFunctionChildRequestsApproval_PropagatesRequestToCallerAsync()
    {
        // Arrange
        var childApproval = new ToolApprovalRequestContent(
            "approval-1",
            new FunctionCallContent("child-call-1", "DeleteFile"));
        var childResponse = new ChatResponse([
            new ChatMessage(ChatRole.Assistant, [childApproval])
        ]);

        Mock<IChatClient> childChatClient = new();
        childChatClient
            .Setup(client => client.GetResponseAsync(
                It.IsAny<IEnumerable<ChatMessage>>(),
                It.IsAny<ChatOptions>(),
                It.IsAny<CancellationToken>()))
            .Callback<IEnumerable<ChatMessage>, ChatOptions, CancellationToken>((messages, _, _) =>
            {
                this.WriteMessages("3. Child model input", messages);
                this.WriteMessages("4. Child model response", childResponse.Messages);
            })
            .ReturnsAsync(childResponse);

        var childAgent = new ChatClientAgent(childChatClient.Object, name: "ChildAgent");
        var childFunction = childAgent.AsAIFunction();
        var parentResponse = new ChatResponse([
            new ChatMessage(ChatRole.Assistant, [
                new FunctionCallContent(
                    "parent-call-1",
                    childFunction.Name,
                    new Dictionary<string, object?>
                    {
                        ["query"] = "Delete a file"
                    })
            ])
        ]);

        Mock<IChatClient> parentChatClient = new();
        parentChatClient
            .Setup(client => client.GetResponseAsync(
                It.IsAny<IEnumerable<ChatMessage>>(),
                It.IsAny<ChatOptions>(),
                It.IsAny<CancellationToken>()))
            .Callback<IEnumerable<ChatMessage>, ChatOptions, CancellationToken>((messages, _, _) =>
            {
                this.WriteMessages("1. Parent model input", messages);
                this.WriteMessages("2. Parent model response", parentResponse.Messages);
            })
            .ReturnsAsync(parentResponse);

        var parentAgent = new ChatClientAgent(parentChatClient.Object, tools: [childFunction]);
        var parentSession = await parentAgent.CreateSessionAsync();

        // Act
        var response = await parentAgent.RunAsync("Delete a file", parentSession);
        this.WriteMessages("5. Parent agent final response", response.Messages);

        // Assert
        Assert.Contains(response.Messages, message =>
            message.Contents.OfType<FunctionCallContent>()
                .Any(call => call.CallId == "parent-call-1"));

        var approval = Assert.Single(response.Messages
            .SelectMany(message => message.Contents)
            .OfType<ToolApprovalRequestContent>());

        Assert.Equal("approval-1", approval.RequestId);

        var childCall = Assert.IsType<FunctionCallContent>(approval.ToolCall);
        Assert.Equal("child-call-1", childCall.CallId);
        Assert.Equal("DeleteFile", childCall.Name);

        Assert.DoesNotContain(
            response.Messages.SelectMany(message => message.Contents),
            content => content is FunctionResultContent);
    }

    private void WriteMessages(string stage, IEnumerable<ChatMessage> messages)
    {
        this._output.WriteLine(stage);

        var messageIndex = 0;
        foreach (var message in messages)
        {
            this._output.WriteLine($"  Message[{messageIndex++}] Role={message.Role}");

            var contentIndex = 0;
            foreach (var content in message.Contents)
            {
                this._output.WriteLine($"    Content[{contentIndex++}] {FormatContent(content)}");
            }
        }
    }

    private static string FormatContent(AIContent content) => content switch
    {
        TextContent text => $"TextContent Text={text.Text}",
        FunctionCallContent call =>
            $"FunctionCallContent CallId={call.CallId}, Name={call.Name}, Arguments={FormatArguments(call.Arguments)}",
        FunctionResultContent result =>
            $"FunctionResultContent CallId={result.CallId}, ResultType={result.Result?.GetType().FullName ?? "null"}, Result={result.Result}",
        ToolApprovalRequestContent approval =>
            $"ToolApprovalRequestContent RequestId={approval.RequestId}, ToolCall=({FormatContent(approval.ToolCall)})",
        _ => $"{content.GetType().FullName}: {content}",
    };

    private static string FormatArguments(IDictionary<string, object?>? arguments) =>
        arguments is null
            ? "null"
            : $"{{{string.Join(", ", arguments.Select(argument => $"{argument.Key}={argument.Value}"))}}}";
}
