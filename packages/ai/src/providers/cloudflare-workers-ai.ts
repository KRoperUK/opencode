import { Effect, Schema } from "effect"
import type { ProviderPackage } from "../provider-package.js"
import { CloudflareRunExtraction } from "../experimental/cloudflare-run-extraction.js"
import { CloudflareRunDecisions } from "../protocols/cloudflare-run-decisions.js"
import { CloudflareWorkersAIImages } from "../protocols/cloudflare-workers-ai-images.js"
import { OpenAIChat } from "../protocols/openai-chat.js"
import { OpenResponses } from "../protocols/open-responses.js"
import { JsonObject, ProviderShared } from "../protocols/shared.js"
import { AuthOptions, type AtLeastOne, type ProviderAuthOption } from "../route/auth-options.js"
import { Route, type RouteDefaultsInput } from "../route/client.js"
import { Endpoint } from "../route/endpoint.js"
import { MediaRoute } from "../route/media.js"
import { Protocol } from "../route/protocol.js"
import { ProviderConfigurationError, ProviderID, type ModelID, type ProviderOptions } from "../schema/index.js"
import type { OpenAIProviderOptionsInput } from "./openai-options.js"
import type { OpenResponsesProviderOptionsInput } from "./open-responses-options.js"

export const id = ProviderID.make("cloudflare-workers-ai")
export const authEnvVars = ["CLOUDFLARE_API_KEY", "CLOUDFLARE_WORKERS_AI_TOKEN", "CLOUDFLARE_API_TOKEN"] as const

type WorkersAIURL = AtLeastOne<{
  readonly accountId: string
  readonly baseURL: string
}>

export type Config = WorkersAIURL &
  Omit<RouteDefaultsInput, "providerOptions"> &
  ProviderAuthOption<"optional"> & {
    readonly providerOptions?: OpenAIProviderOptionsInput | OpenResponsesProviderOptionsInput
  }

export type Settings<Options = OpenAIProviderOptionsInput> = ProviderPackage.Settings &
  Options &
  WorkersAIURL & {
    readonly apiKey?: string | false
  }

export type { CloudflareWorkersAIImageOptions } from "../protocols/cloudflare-workers-ai-images.js"

export const baseURL = (input: WorkersAIURL) => {
  if (input.baseURL) return input.baseURL
  if (!input.accountId)
    throw new ProviderConfigurationError({
      provider: id,
      message: "CloudflareWorkersAI.configure requires accountId unless baseURL is supplied",
    })
  return `https://api.cloudflare.com/client/v4/accounts/${encodeURIComponent(input.accountId)}/ai/v1`
}

/** The Run API (`/ai/run`) next to the OpenAI-compatible `/ai/v1` base. */
export const runBaseURL = (input: WorkersAIURL) => {
  const url = Endpoint.trimBaseUrl(baseURL(input))
  if (!url.endsWith("/ai/v1"))
    throw new ProviderConfigurationError({
      provider: id,
      message: "CloudflareWorkersAI needs accountId or a baseURL ending in /ai/v1 to derive the Run API base",
    })
  return `${url.slice(0, -"/v1".length)}/run`
}

export const route = Route.make({
  id: "cloudflare-workers-ai",
  provider: id,
  providerMetadataKey: "cloudflare-workers-ai",
  protocol: OpenAIChat.protocol,
  endpoint: Endpoint.path("/chat/completions"),
  framing: OpenAIChat.framing,
})

const ResponsesBody = Schema.Struct({
  ...OpenResponses.OpenResponsesBody.fields,
  stream: Schema.Literal(false),
  input: Schema.Array(JsonObject),
  tools: Schema.optional(Schema.Array(JsonObject)),
})

// Workers AI answers `/responses` only with the complete Response, not a stream.
const responsesProtocol = Protocol.make({
  id: "cloudflare-workers-ai-responses",
  body: {
    schema: ResponsesBody,
    from: (request) => OpenResponses.fromRequest(request).pipe(Effect.flatMap(responsesBody)),
  },
  stream: OpenResponses.protocol.stream,
})

const responsesRoute = Route.make({
  id: "cloudflare-workers-ai-responses",
  provider: id,
  providerMetadataKey: "cloudflare-workers-ai",
  protocol: responsesProtocol,
  endpoint: Endpoint.path("/responses"),
  framing: OpenResponses.documentFraming,
  defaults: { providerOptions: { store: false } },
})

export const routes = [route, responsesRoute]

export const configure = (input: Config) => {
  const { accountId: _accountId, apiKey: _apiKey, auth: _auth, baseURL: _baseURL, ...defaults } = input
  const auth = AuthOptions.bearer(input, authEnvVars)
  const options = { ...defaults, endpoint: { baseURL: baseURL(input) }, auth }
  const configured = route.with(options)
  const configuredResponses = responsesRoute.with(options)
  // The Run API base is derived per call so a chat-only configuration without an account still constructs.
  const runDeployment = () => MediaRoute.deployment({ ...input, baseURL: runBaseURL(input) }, auth)
  const decision = (modelID: string | ModelID, providerOptions?: ProviderOptions) =>
    CloudflareRunDecisions.route
      .with({
        provider: id,
        endpoint: { baseURL: runBaseURL(input) },
        auth: auth,
        headers: input.headers,
        http: input.http,
      })
      .model({ id: modelID, providerOptions })
  return {
    id,
    model: (modelID: string | ModelID) => configured.model<OpenAIProviderOptionsInput>({ id: modelID }),
    responses: (modelID: string | ModelID) =>
      configuredResponses.model<OpenResponsesProviderOptionsInput>({ id: modelID }),
    decision,
    image: (modelID: string | ModelID, providerOptions?: ProviderOptions) =>
      CloudflareWorkersAIImages.model({ ...runDeployment(), id: modelID, providerOptions }),
    experimental: {
      extraction: (modelID: string | ModelID) => CloudflareRunExtraction.model({ ...runDeployment(), id: modelID }),
    },
    configure,
  }
}

export const provider = { id, configure }

export const model: ProviderPackage.Model<Settings, OpenAIProviderOptionsInput> = (modelID, settings) =>
  configure(config(settings)).model(modelID)
export const responsesModel: ProviderPackage.Model<
  Settings<OpenResponsesProviderOptionsInput>,
  OpenResponsesProviderOptionsInput
> = (modelID, settings) => configure(config(settings)).responses(modelID)
export const imageModel = (modelID: string, settings: Settings) => {
  const input = config(settings)
  return configure(input).image(modelID, input.providerOptions)
}
export const decisionModel = (modelID: string, settings: Settings) => {
  const input = config(settings)
  return configure(input).decision(modelID, input.providerOptions)
}

// Workers AI accepts replayed input items only in the shape of its own output items, and gpt-oss-120b reads a tool's
// definition from the Chat Completions `function` member. Forced tool choice fails upstream on both gpt-oss models.
const responsesBody = (body: OpenResponses.OpenResponsesBody) => {
  if (body.tool_choice !== undefined && body.tool_choice !== "auto" && body.tool_choice !== "none")
    return Effect.fail(ProviderShared.invalidRequest("Workers AI Responses supports only automatic tool choice"))
  const tools = body.tool_choice === "none" ? undefined : body.tools
  return Effect.succeed({
    ...body,
    stream: false as const,
    tool_choice: undefined,
    input: body.input.flatMap(responsesInputItem),
    tools: tools?.map((tool) =>
      tool.type === "function"
        ? { ...tool, function: { name: tool.name, description: tool.description, parameters: tool.parameters } }
        : tool,
    ),
  })
}

type InputItem = OpenResponses.OpenResponsesBody["input"][number]

const responsesInputItem = (item: InputItem, index: number): ReadonlyArray<Record<string, unknown>> => {
  if (isItem(item, "function_call")) return [{ ...item, status: null }]
  if (isItem(item, "reasoning")) {
    if (item.id === undefined) return []
    return [{ ...item, content: item.summary.map((part) => ({ type: "reasoning_text", text: part.text })) }]
  }
  // Tool output must be a string; gpt-oss reads text only.
  if (isItem(item, "function_call_output", "custom_tool_call_output") && typeof item.output !== "string")
    return [
      { ...item, output: item.output.flatMap((part) => (part.type === "input_text" ? [part.text] : [])).join("\n") },
    ]
  if (isItem(item, "message") && item.role === "assistant")
    return [
      {
        ...item,
        id: item.id ?? `msg_${index}`,
        content: item.content.map((part) => ({ ...part, annotations: [] })),
      },
    ]
  return [item]
}

// Hosted tool replays carry an open `type`, so `item.type` alone does not narrow the input union.
const isItem = <const Type extends string>(
  item: InputItem,
  ...types: ReadonlyArray<Type>
): item is Extract<InputItem, { readonly type: Type }> => types.some((type) => type === item.type)

function config(settings: Settings<Config["providerOptions"]>) {
  const { accountId: _, apiKey, baseURL: _url, body, headers, ...providerOptions } = settings
  return {
    apiKey,
    baseURL: baseURL(settings),
    headers,
    http: body === undefined ? undefined : { body },
    providerOptions,
  }
}

export * as CloudflareWorkersAI from "./cloudflare-workers-ai.js"
