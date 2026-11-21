import { DurableObject } from "cloudflare:workers";

interface Env {
	ARES_CONTAINER: DurableObjectNamespace<AresNycContainer>;
	GEMINI_API_KEY: string;
}

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
			container.start({
				image: container.images.base,
				enableInternet: true,
				env: {
					GEMINI_API_KEY: this.env.GEMINI_API_KEY,
					PYTHONUNBUFFERED: "1",
				},
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
		return env.ARES_CONTAINER.getByName("nyc-live-main").fetch(request);
	},
};
