/* AudioWorklet: microphone audio in, 16 kHz mono PCM16 little-endian frames out.
 *
 * The browser's sample rate (usually 44.1 or 48 kHz) is reduced by averaging each group of
 * input samples (a simple low-pass followed by decimation). Every frameMs of audio it posts
 *   { pcm: ArrayBuffer }  (Int16 samples, transferred)
 * and about every 40 ms it posts
 *   { level: number }     (RMS of the last 40 ms, 0..1, for the level meter)
 */
class Pcm16Downsampler extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const o = (options && options.processorOptions) || {};
    this.target = o.targetRate || 16000;
    this.frameSamples = Math.round((this.target * (o.frameMs || 100)) / 1000);
    this.levelSamples = Math.round(this.target * 0.04);
    this.ratio = sampleRate / this.target;
    this.buf = new Int16Array(this.frameSamples);
    this.n = 0;
    this.acc = 0;
    this.accN = 0;
    this.phase = 0;
    this.sumSq = 0;
    this.levelN = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.acc += ch[i];
      this.accN += 1;
      this.phase += 1;
      if (this.phase < this.ratio) continue;
      this.phase -= this.ratio;
      const s = Math.max(-1, Math.min(1, this.acc / this.accN));
      this.acc = 0;
      this.accN = 0;
      this.buf[this.n++] = s < 0 ? s * 0x8000 : s * 0x7fff;
      this.sumSq += s * s;
      this.levelN += 1;
      if (this.levelN >= this.levelSamples) {
        this.port.postMessage({ level: Math.sqrt(this.sumSq / this.levelN) });
        this.sumSq = 0;
        this.levelN = 0;
      }
      if (this.n >= this.frameSamples) {
        const out = this.buf.buffer;
        this.buf = new Int16Array(this.frameSamples);
        this.n = 0;
        this.port.postMessage({ pcm: out }, [out]);
      }
    }
    return true;
  }
}

registerProcessor("pcm16-downsampler", Pcm16Downsampler);
