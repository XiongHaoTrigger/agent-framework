// Copyright (c) Microsoft. All rights reserved.

using System.Text.Json.Serialization;

namespace Microsoft.Agents.AI.ChatClient;

/// <summary>
/// Just tips parent agent this is approval result from child agent when child agent is a function.
/// </summary>
internal sealed class AgentAsFunctionResult
{
    [JsonPropertyName("continuationId")]
    public string ContinuationId { get; init; } = string.Empty;
}
