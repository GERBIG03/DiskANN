#include "adaptive_reverse_prune.h"

namespace diskann
{

template class AdaptiveReversePruner<float>;
template class AdaptiveReversePruner<int8_t>;
template class AdaptiveReversePruner<uint8_t>;

} // namespace diskann
