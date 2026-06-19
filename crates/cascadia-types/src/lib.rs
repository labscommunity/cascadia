//! Engine-agnostic core types.
//!
//! Mirrors `cascadia/shared/types.py`, `cascadia/shared/shard.py`, and the
//! peer-layout types from `cascadia/shared/topology.py`. The richer
//! topology graph lives in `cascadia-topology` (it carries state +
//! behaviour).

pub mod chunk;
pub mod error;
pub mod load;
pub mod peer;
pub mod shard;
pub mod task;

pub use chunk::{Chunk, TokenLogprobs, TopLogprob};
pub use error::Error;
pub use load::LoadProgress;
pub use peer::{PeerEndpoint, PeerLayout};
pub use shard::{ShardPlan, ShardSpec};
pub use task::{GenerationTask, TaskId};
