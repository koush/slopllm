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
        if (this._value === value)
            return;
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

export class DisposableSet implements Disposable {
    private _items: Disposable[] = [];

    add(value: Disposable) {
        this._items.push(value);
    }

    [Symbol.dispose]() {
        const items = this._items;
        this._items = [];
        for (const item of items) {
            item[Symbol.dispose]();
        }
    }
}

export class UsingHolderMap<K, T extends Disposable> implements Disposable {
    private map: Map<K, UsingHolder<T>> = new Map();

    set(key: K, value: T) {
        let existing = this.map.get(key);
        if (!existing) {
            existing = new UsingHolder(undefined!);
            this.map.set(key, existing);
        }
        existing.replace(value);
    }

    get(key: K): T | undefined {
        return this.map.get(key)?.value;
    }

    detach(key: K): T | undefined {
        const holder = this.map.get(key);
        if (!holder) {
            return undefined;
        }
        this.map.delete(key);
        return holder.detach();
    }

    *entries(): IterableIterator<[K, T]> {
        for (const [key, holder] of this.map) {
            yield [key, holder.value];
        }
    }

    detachAll(): Map<K, T> {
        const result = new Map<K, T>();
        for (const [key, holder] of this.map) {
            result.set(key, holder.detach());
        }
        this.map.clear();
        return result;
    }

    [Symbol.dispose]() {
        for (const value of this.map.values()) {
            value[Symbol.dispose]();
        }
        this.map.clear();
    }
}
