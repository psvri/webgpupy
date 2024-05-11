pub enum Holder<'a, B> {
    Borrowed(&'a B),
    Owned(B),
}

impl<'a, T> AsRef<T> for Holder<'a, T> {
    fn as_ref(&self) -> &T {
        match self {
            Holder::Borrowed(x) => x,
            Holder::Owned(x) => x,
        }
    }
}
