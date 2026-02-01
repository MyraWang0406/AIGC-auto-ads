/**
 * Cloudflare Worker：入口，将 HTTP 请求代理到 Streamlit 容器。
 * /healthz 直接返回 ok，不进容器；其余请求代理到固定容器实例。
 */
import { Container, getContainer } from "@cloudflare/containers";
import { env } from "cloudflare:workers";

export class StreamlitContainer extends Container {
  defaultPort = 8080;
  sleepAfter = "10m";

  /** Worker Secrets 传给容器（wrangler secret put OPENROUTER_API_KEY 等） */
  envVars = {
    OPENROUTER_API_KEY: (env as Record<string, string>).OPENROUTER_API_KEY ?? "",
    OPENROUTER_MODEL: (env as Record<string, string>).OPENROUTER_MODEL ?? "openai/gpt-4o-mini",
  };

  override onStart() {
    console.log("Streamlit container started");
  }

  override onStop() {
    console.log("Streamlit container stopped");
  }

  override onError(error: unknown) {
    console.error("Streamlit container error:", error);
  }
}

export default {
  async fetch(
    request: Request,
    env: { STREAMLIT_CONTAINER: DurableObjectNamespace }
  ): Promise<Response> {
    const url = new URL(request.url);
    const pathname = url.pathname;

    // /healthz 不进入容器，直接返回
    if (pathname === "/healthz") {
      return new Response(JSON.stringify({ status: "ok" }), {
        headers: { "Content-Type": "application/json" },
      });
    }

    // 其余请求代理到固定容器实例（id=1，单例模式）
    const container = getContainer(env.STREAMLIT_CONTAINER, "streamlit-app");
    return container.fetch(request);
  },
};
