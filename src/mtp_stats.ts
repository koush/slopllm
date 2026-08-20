/** Host-side acceptance metrics for speculative MTP decoding. */
export class MtpStats {
  numDrafts = 0;
  numDraftTokens = 0;
  numAcceptedTokens = 0;
  numAcceptedPerPos: number[];
  readonly numSpecTokens: number;

  constructor(numSpecTokens: number) {
    this.numSpecTokens = numSpecTokens;
    this.numAcceptedPerPos = new Array(numSpecTokens).fill(0);
  }

  observe(numDraftTokens: number, numAccepted: number): void {
    this.numDrafts++;
    this.numDraftTokens += numDraftTokens;
    this.numAcceptedTokens += numAccepted;
    for (let i = 0; i < numAccepted && i < this.numAcceptedPerPos.length; i++) {
      this.numAcceptedPerPos[i]++;
    }
  }

  get acceptanceRate(): number {
    return this.numDraftTokens > 0 ? this.numAcceptedTokens / this.numDraftTokens : NaN;
  }

  get meanAcceptanceLength(): number {
    return this.numDrafts > 0 ? 1 + this.numAcceptedTokens / this.numDrafts : NaN;
  }

  perPositionRates(): number[] {
    return this.numAcceptedPerPos.map(accepted => this.numDrafts > 0 ? accepted / this.numDrafts : NaN);
  }

  log(): string {
    if (this.numDrafts === 0) {
      return "";
    }
    const rates = this.perPositionRates().map(rate => rate.toFixed(3)).join(", ");
    return `MTP metrics: mean acceptance length=${this.meanAcceptanceLength.toFixed(2)}, ` +
      `acceptance rate=${(this.acceptanceRate * 100).toFixed(1)}%, ` +
      `accepted=${this.numAcceptedTokens}/${this.numDraftTokens} tokens, ` +
      `drafts=${this.numDrafts}, per-pos=[${rates}]`;
  }
}
