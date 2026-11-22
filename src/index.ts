import { DurableObject } from "cloudflare:workers";

interface Env {
	ARES_CONTAINER: DurableObjectNamespace<AresNycContainer>;
	CF_VERSION_METADATA: { id: string };
	GEMINI_API_KEY?: string;
	GEMINI_API_KEYS?: string;
	GEMINI_API_KEY_FALLBACK?: string;
	GEMINI_API_KEY_2?: string;
	GEMINI_API_KEY_3?: string;
	GOOGLE_API_KEY?: string;
	OPENAI_API_KEY?: string;
	ANTHROPIC_API_KEY?: string;
	OPENROUTER_API_KEY?: string;
}

const FORWARDED_SECRET_KEYS = [
	"GEMINI_API_KEY",
	"GEMINI_API_KEYS",
	"GEMINI_API_KEY_FALLBACK",
	"GEMINI_API_KEY_2",
	"GEMINI_API_KEY_3",
	"GOOGLE_API_KEY",
	"OPENAI_API_KEY",
	"ANTHROPIC_API_KEY",
	"OPENROUTER_API_KEY",
] as const;

const CONTAINER_PORT = 8080;
const INACTIVITY_TIMEOUT_MS = 10 * 60 * 1000;

function rewriteRequestForContainer(request: Request): Request {
	const url = new URL(request.url);
	url.protocol = "http:";
	url.host = "container";
	const forwarded = new Request(url, request);
	forwarded.headers.delete("host");
	return forwarded;
}

export class AresNycContainer extends DurableObject<Env> {
	private async ensureReady(): Promise<void> {
		const container = this.ctx.container;
		if (!container) {
			throw new Error("Container binding is not available on this Durable Object");
		}
		if (!container.running) {
			const containerEnv: Record<string, string> = {
				PYTHONUNBUFFERED: "1",
			};
			for (const key of FORWARDED_SECRET_KEYS) {
				const value = this.env[key];
				if (value) {
					containerEnv[key] = value;
				}
			}
			container.start({
				image: container.images.base,
				instance: "basic",
				enableInternet: true,
				env: containerEnv,
			});
		}
		await container.setInactivityTimeout(INACTIVITY_TIMEOUT_MS);

		const port = container.getTcpPort(CONTAINER_PORT);
		let lastError: unknown;
		for (let attempt = 0; attempt < 100; attempt++) {
			try {
				const response = await port.fetch("http://container/health", {
					signal: AbortSignal.timeout(1000),
				});
				await response.body?.cancel();
				if (response.ok) {
					return;
				}
				lastError = new Error(`Health check returned ${response.status}`);
			} catch (error) {
				lastError = error;
			}
			await scheduler.wait(200);
		}
		throw new Error(`Container did not become ready on port ${CONTAINER_PORT}`, {
			cause: lastError,
		});
	}

	async fetch(request: Request): Promise<Response> {
		await this.ensureReady();
		const forwarded = rewriteRequestForContainer(request);
		return this.ctx.container!.getTcpPort(CONTAINER_PORT).fetch(forwarded);
	}
}

export default {
	async fetch(request: Request, env: Env): Promise<Response> {
		const instanceName = `nyc-live-${env.CF_VERSION_METADATA?.id || "main"}`;
		return env.ARES_CONTAINER.getByName(instanceName).fetch(request);
	},
};
