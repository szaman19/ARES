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

// Site-wide + per-IP sliding-window rate limits (enforced in the single Durable Object)
const WINDOW_MS = 60_000;
const GLOBAL_SITE_LIMIT = 120; // max 120 requests/min across entire site
const IP_SITE_LIMIT = 60; // max 60 requests/min per IP
const GLOBAL_SEARCH_LIMIT = 6; // max 6 live searches/min across entire site
const IP_SEARCH_LIMIT = 3; // max 3 live searches/min per IP

function pruneAndCheck(timestamps: number[], now: number, limit: number): { allowed: boolean; retryAfterS: number } {
	const cutoff = now - WINDOW_MS;
	while (timestamps.length > 0 && timestamps[0] <= cutoff) {
		timestamps.shift();
	}
	if (timestamps.length >= limit) {
		const retryAfterS = Math.max(1, Math.ceil((timestamps[0] + WINDOW_MS - now) / 1000));
		return { allowed: false, retryAfterS };
	}
	timestamps.push(now);
	return { allowed: true, retryAfterS: 0 };
}

function rateLimitResponse(message: string, retryAfterS: number): Response {
	return new Response(
		JSON.stringify({ error: message, retry_after_s: retryAfterS }),
		{
			status: 429,
			headers: {
				"content-type": "application/json",
				"retry-after": String(retryAfterS),
				"cache-control": "no-store",
			},
		},
	);
}

function rewriteRequestForContainer(request: Request): Request {
	const url = new URL(request.url);
	url.protocol = "http:";
	url.host = "container";
	const forwarded = new Request(url, request);
	forwarded.headers.delete("host");
	return forwarded;
}

export class AresNycContainer extends DurableObject<Env> {
	private globalSiteHits: number[] = [];
	private globalSearchHits: number[] = [];
	private ipSiteHits = new Map<string, number[]>();
	private ipSearchHits = new Map<string, number[]>();

	private checkRateLimit(request: Request): Response | null {
		const url = new URL(request.url);
		const path = url.pathname;
		if (path === "/health") {
			return null;
		}
		const now = Date.now();
		const ip = request.headers.get("cf-connecting-ip") || request.headers.get("x-forwarded-for") || "unknown";

		// Periodically bound per-IP map size
		if (this.ipSiteHits.size > 2000) {
			this.ipSiteHits.clear();
		}
		if (this.ipSearchHits.size > 2000) {
			this.ipSearchHits.clear();
		}

		// 1. Global + per-IP site-wide cap
		const globalSite = pruneAndCheck(this.globalSiteHits, now, GLOBAL_SITE_LIMIT);
		if (!globalSite.allowed) {
			return rateLimitResponse("Site-wide rate limit reached. Please retry shortly.", globalSite.retryAfterS);
		}
		let ipSite = this.ipSiteHits.get(ip);
		if (!ipSite) {
			ipSite = [];
			this.ipSiteHits.set(ip, ipSite);
		}
		const ipSiteCheck = pruneAndCheck(ipSite, now, IP_SITE_LIMIT);
		if (!ipSiteCheck.allowed) {
			return rateLimitResponse("Too many requests from your IP. Please slow down.", ipSiteCheck.retryAfterS);
		}

		// 2. Stricter cap on expensive search / ablation endpoints
		const isExpensive =
			path === "/api/search" ||
			path === "/api/search/stream" ||
			path === "/api/ablate";
		if (isExpensive) {
			const globalSearch = pruneAndCheck(this.globalSearchHits, now, GLOBAL_SEARCH_LIMIT);
			if (!globalSearch.allowed) {
				return rateLimitResponse(
					"Site-wide search rate limit reached (protecting NYC DOT cameras & API quota). Try again in a moment.",
					globalSearch.retryAfterS,
				);
			}
			let ipSearch = this.ipSearchHits.get(ip);
			if (!ipSearch) {
				ipSearch = [];
				this.ipSearchHits.set(ip, ipSearch);
			}
			const ipSearchCheck = pruneAndCheck(ipSearch, now, IP_SEARCH_LIMIT);
			if (!ipSearchCheck.allowed) {
				return rateLimitResponse(
					"Too many live searches from your IP. Please wait before searching again.",
					ipSearchCheck.retryAfterS,
				);
			}
		}

		return null;
	}

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
		const blocked = this.checkRateLimit(request);
		if (blocked) {
			return blocked;
		}
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
