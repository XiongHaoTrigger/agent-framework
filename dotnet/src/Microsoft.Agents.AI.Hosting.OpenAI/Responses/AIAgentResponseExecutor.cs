// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Agents.AI.Hosting.OpenAI.Responses.Models;
using Microsoft.Agents.AI.Hosting.OpenAI.ChatCompletions.Models;
using Microsoft.Extensions.AI;

namespace Microsoft.Agents.AI.Hosting.OpenAI.Responses;

/// <summary>
/// Response executor that uses an AIAgent to execute responses locally.
/// This is the default implementation for local execution.
/// </summary>
internal sealed class AIAgentResponseExecutor : IResponseExecutor
{
    private readonly AIAgent _agent;

    public AIAgentResponseExecutor(AIAgent agent)
    {
        ArgumentNullException.ThrowIfNull(agent);
        this._agent = agent;
    }

    public ValueTask<ResponseError?> ValidateRequestAsync(
        CreateResponse request,
        CancellationToken cancellationToken = default) => ValueTask.FromResult<ResponseError?>(null);

    public async IAsyncEnumerable<StreamingResponseEvent> ExecuteAsync(
        AgentInvocationContext context,
        CreateResponse request,
        IReadOnlyList<ChatMessage>? conversationHistory = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var options = request.ToChatClientAgentRunOptions();

        // Convert input to chat messages, prepending conversation history if available
        var messages = new List<ChatMessage>();

        if (conversationHistory is not null)
        {
            messages.AddRange(conversationHistory);
        }

        foreach (var inputMessage in request.Input.GetInputMessages())
        {
            messages.Add(inputMessage.ToChatMessage());
        }

        // Use the extension method to convert streaming updates to streaming response events
        await foreach (var streamingEvent in this._agent.RunStreamingAsync(messages, options: options, cancellationToken: cancellationToken)
            .ToStreamingResponseAsync(request, context, cancellationToken)
            .ConfigureAwait(false))
        {
            yield return streamingEvent;
        }
    }
}

internal static class CreateResponseChatClientAgentRunOptionsExtensions
{
    private static readonly JsonElement s_emptyJson = JsonElement.Parse("{}");

    public static ChatClientAgentRunOptions ToChatClientAgentRunOptions(this CreateResponse request)
    {
        ChatOptions chatOptions = new()
        {
            // Note: We intentionally do NOT set ConversationId on ChatOptions here.
            // The conversation ID from the client request is used by the hosting layer
            // to manage conversation storage, but should not be forwarded to the underlying
            // IChatClient as it has its own concept of conversations (or none at all).
            // ---
            // ConversationId = request.Conversation?.Id,
            Temperature = (float?)request.Temperature,
            TopP = (float?)request.TopP,
            MaxOutputTokens = request.MaxOutputTokens,
            Instructions = request.Instructions,
            ModelId = request.Model,
            AllowMultipleToolCalls = request.ParallelToolCalls,
        };

        if (request.ToolChoice is { } toolChoice)
        {
            chatOptions.ToolMode = toolChoice.ToChatToolMode();
        }

        if (request.Tools is { Count: > 0 })
        {
            List<AITool> tools = [];
            foreach (JsonElement tool in request.Tools)
            {
                if (tool.ToAITool() is { } aiTool)
                {
                    tools.Add(aiTool);
                }
            }

            if (tools.Count > 0)
            {
                chatOptions.Tools = tools;
            }
        }

        return new ChatClientAgentRunOptions(chatOptions);
    }

    private static AITool? ToAITool(this JsonElement tool)
    {
        if (tool.ValueKind != JsonValueKind.Object ||
            !tool.TryGetProperty("type", out JsonElement typeProperty) ||
            typeProperty.GetString() is not { } type)
        {
            return null;
        }

        if (string.Equals(type, "function", StringComparison.Ordinal))
        {
            return ToAIFunction(tool);
        }

        if (string.Equals(type, "custom", StringComparison.Ordinal) &&
            TryGetToolName(tool, out string? name))
        {
            return new CustomAITool(name, TryGetStringProperty(tool, "description"), additionalProperties: null);
        }

        return null;
    }

    private static AIFunctionDeclaration? ToAIFunction(JsonElement tool)
    {
        JsonElement function = tool.TryGetProperty("function", out JsonElement functionProperty) &&
            functionProperty.ValueKind == JsonValueKind.Object
            ? functionProperty
            : tool;

        if (!TryGetToolName(function, out string? name))
        {
            return null;
        }

        JsonElement parameters = function.TryGetProperty("parameters", out JsonElement parametersProperty)
            ? parametersProperty
            : s_emptyJson;

        return AIFunctionFactory.CreateDeclaration(
            name,
            TryGetStringProperty(function, "description"),
            parameters);
    }

    private static ChatToolMode? ToChatToolMode(this JsonElement toolChoice)
    {
        if (toolChoice.ValueKind == JsonValueKind.String)
        {
            return toolChoice.GetString() switch
            {
                "auto" => ChatToolMode.Auto,
                "none" => ChatToolMode.None,
                "required" => ChatToolMode.RequireAny,
                _ => null
            };
        }

        if (toolChoice.ValueKind != JsonValueKind.Object ||
            !toolChoice.TryGetProperty("type", out JsonElement typeProperty) ||
            typeProperty.GetString() is not { } type)
        {
            return null;
        }

        if (string.Equals(type, "allowed_tools", StringComparison.Ordinal) &&
            toolChoice.TryGetProperty("allowed_tools", out JsonElement allowedTools) &&
            allowedTools.TryGetProperty("mode", out JsonElement modeProperty))
        {
            return modeProperty.GetString() switch
            {
                "auto" => ChatToolMode.Auto,
                "required" => ChatToolMode.RequireAny,
                _ => null
            };
        }

        return TryGetToolName(toolChoice, out string? name) ? ChatToolMode.RequireSpecific(name) : null;
    }

    private static bool TryGetToolName(JsonElement element, out string name)
    {
        if (element.TryGetProperty("name", out JsonElement nameProperty) &&
            nameProperty.ValueKind == JsonValueKind.String &&
            nameProperty.GetString() is { } directName)
        {
            name = directName;
            return true;
        }

        if (element.TryGetProperty("function", out JsonElement functionProperty) &&
            functionProperty.ValueKind == JsonValueKind.Object &&
            functionProperty.TryGetProperty("name", out JsonElement functionNameProperty) &&
            functionNameProperty.ValueKind == JsonValueKind.String &&
            functionNameProperty.GetString() is { } functionName)
        {
            name = functionName;
            return true;
        }

        if (element.TryGetProperty("custom", out JsonElement customProperty) &&
            customProperty.ValueKind == JsonValueKind.Object &&
            customProperty.TryGetProperty("name", out JsonElement customNameProperty) &&
            customNameProperty.ValueKind == JsonValueKind.String &&
            customNameProperty.GetString() is { } customName)
        {
            name = customName;
            return true;
        }

        name = string.Empty;
        return false;
    }

    private static string? TryGetStringProperty(JsonElement element, string propertyName)
        => element.TryGetProperty(propertyName, out JsonElement property) &&
            property.ValueKind == JsonValueKind.String
            ? property.GetString()
            : null;
}
