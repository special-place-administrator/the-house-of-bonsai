import { PRISM_DEBUG_LOGGING } from "../config.js";

/**
 * Logs a message to stderr only if PRISM_DEBUG_LOGGING is true.
 * Use this for verbose traces (e.g., initialization, request tracking)
 * that should be hidden from users by default but remain available
 * for troubleshooting.
 */
export function debugLog(message: string) {
  if (PRISM_DEBUG_LOGGING) {
    console.error(message);
  }
}
