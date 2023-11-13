pub type NdArrayResult<T> = Result<T, NdArrayError>;

#[derive(Debug)]
pub enum NdArrayError {
    BroadcastError(String),
    RepeatError(String),
}
