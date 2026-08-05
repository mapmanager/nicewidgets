/**
 * Dataset-scoped cache for decoded, display-oriented channel planes.
 *
 * Entries intentionally remain for the lifetime of one loaded dataset because
 * the viewer targets in-memory scientific datasets that fit browser memory.
 * `RasterViewer.load()` calls `clear()` before every dataset replacement, which
 * aborts in-flight fetches and makes the prior dataset collectible.
 */

import {transposePlane} from './orientation.js';

export class PlaneCache {
  /**
   * Create one cache whose lifetime matches one loaded descriptor.
   *
   * @param {object} descriptor Active versioned raster descriptor.
   * @param {Function|null} [onMetric=null] Optional completed-fetch metric callback.
   */
  constructor(descriptor, onMetric = null) {
    this.descriptor = descriptor;
    this.onMetric = onMetric;
    this.entries = new Map();
    this.controller = new AbortController();
  }

  /** Return the dataset-local identity for one decoded plane or projection. */
  key(channelId, selection = {}) {
    return `${channelId}|t:${selection.t_index ?? 'plane'}|z:${selection.z_index ?? 'plane'}|r:${selection.plus_minus_z ?? 0}`;
  }

  /** Return one shared pending or completed decoded plane promise. */
  get(channel, selection = {}) {
    const key = this.key(channel.id, selection);
    if (!this.entries.has(key)) {
      const pending = this.fetch(channel, selection)
        .catch(error => {
          this.entries.delete(key);
          throw error;
        });
      this.entries.set(key, pending);
    }
    return this.entries.get(key);
  }

  /** Return whether a matching request or plane is already cached. */
  has(channel, selection = {}) {
    return this.entries.has(this.key(channel.id, selection));
  }

  /** Fetch, validate, decode, and transpose one requested channel plane. */
  async fetch(channel, selection = {}) {
    const started = performance.now();
    const url = new URL(channel.data_url, window.location.href);
    if (selection.t_index !== null && selection.t_index !== undefined) {
      url.searchParams.set('t_index', String(selection.t_index));
    }
    if (selection.z_index !== null && selection.z_index !== undefined) {
      url.searchParams.set('z_index', String(selection.z_index));
      url.searchParams.set('plus_minus_z', String(selection.plus_minus_z ?? 0));
    }
    const response = await fetch(url, {cache: 'no-store', signal: this.controller.signal});
    const headersReceived = performance.now();
    if (!response.ok) throw new Error(`channel plane fetch failed: ${response.status}`);
    const buffer = await response.arrayBuffer();
    const bodyReceived = performance.now();
    if (buffer.byteLength !== channel.byte_length) {
      throw new Error('channel plane byte_length mismatch');
    }
    const source = channel.dtype === 'uint16'
      ? new Uint16Array(buffer)
      : new Float32Array(buffer);
    const {height, width} = this.descriptor;
    if (source.length !== width * height) throw new Error('channel plane sample count mismatch');
    const plane = transposePlane(source, height, width);
    const completed = performance.now();
    this.onMetric?.({
      channel_id: channel.id,
      t_index: selection.t_index ?? null,
      z_index: selection.z_index ?? null,
      plus_minus_z: selection.plus_minus_z ?? 0,
      byte_length: buffer.byteLength,
      fetch_headers_ms: headersReceived - started,
      response_body_ms: bodyReceived - headersReceived,
      transpose_ms: completed - bodyReceived,
      total_ms: completed - started,
    });
    return plane;
  }

  /** Abort pending requests and release every dataset-scoped entry. */
  clear() {
    this.controller.abort();
    this.entries.clear();
  }
}
