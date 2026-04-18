export abstract class UsingHolderBase<T> {
    constructor(private _value: T) {
    }

    get value(): T {
        return this._value;
    }

    [Symbol.dispose]() {
        this.release();
    }

    abstract dispose(value: T): void;

    detach() {
        const value = this._value;
        this._value = undefined!;
        return value;
    }

    replace(value: T) {
        this.release();
        this._value = value;
    }

    release() {
        const released = this.detach();
        if (released)
            this.dispose(released);
    }
}

export class UsingHolder<T extends Disposable> extends UsingHolderBase<T> {
    dispose(value: T) {
        value?.[Symbol.dispose]();
    }

    transferClosure<V>(closure: (value: UsingHolder<T>) => Promise<V>) {
        return (async () => {
            using attached = new UsingHolder(this.detach());
            return await closure(attached);
        })();
    }
}

