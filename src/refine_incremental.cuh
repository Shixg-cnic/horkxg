#pragma once
// Included inside namespace gpart after RefineDeviceContext and gain validation.
// Jet supplies the exploration / priority-correction reference. This CUDA
// implementation uses a dense affinity cache for small k, fused exact cut/cache
// updates, and source-quota plus destination-capacity GPU rebalancing.
namespace {
template <typename OffsetT, typename VertexT, typename WeightT>
__global__ void incremental_build_affinity(
    std::int64_t n, const OffsetT* offsets, const VertexT* neighbors,
    const WeightT* weights, const VertexT* labels, int k, std::uint64_t* cache) {
    const int lane=threadIdx.x&31;
    const int warp=threadIdx.x/32;
    const auto v=static_cast<std::int64_t>(blockIdx.x)*8+warp;
    if (v>=n) return;
    __shared__ unsigned long long values[8*32];
    auto* row=values+warp*32;
    row[lane]=0;
    __syncwarp();
    for (std::int64_t e=static_cast<std::int64_t>(offsets[v])+lane; e<offsets[v+1]; e+=32) {
        const auto u=neighbors[e];
        if (u!=v) atomicAdd(row+labels[u],static_cast<unsigned long long>(weights[e]));
    }
    __syncwarp();
    if (lane<k) cache[v*k+lane]=row[lane];
}

template <typename VertexT>
__global__ void incremental_lp_proposals(
    std::int64_t n, const std::uint64_t* cache, const VertexT* labels, int k,
    double filter, std::int32_t* locks, std::int32_t* targets, std::int64_t* gains) {
    const auto v=static_cast<std::int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (v>=n) return;
    targets[v]=-1;gains[v]=INT64_MIN;
    if (locks[v]) { locks[v]=0;return; }
    const auto source=labels[v];
    std::uint64_t best=0;
    int target=-1;
    for (int p=0;p<k;++p) {
        if (p!=source && cache[v*k+p]>best) { best=cache[v*k+p];target=p; }
    }
    const auto current=cache[v*k+source];
    if (target>=0 && (best>=current || current-best<static_cast<std::uint64_t>(filter*current))) {
        targets[v]=target;
        gains[v]=best>=current ? static_cast<std::int64_t>(best-current) : -static_cast<std::int64_t>(current-best);
    }
}

template <typename OffsetT, typename VertexT, typename WeightT>
__global__ void incremental_priority_filter(
    std::int64_t n,const OffsetT* offsets,const VertexT* neighbors,const WeightT* weights,
    const VertexT* labels,const std::int32_t* targets,const std::int64_t* gains,std::int32_t* flags) {
    const int lane=threadIdx.x&31;
    const auto v=static_cast<std::int64_t>(blockIdx.x)*8+threadIdx.x/32;
    if (v>=n) return;
    if (targets[v]<0) { if (!lane) flags[v]=0;return; }
    const auto source=labels[v],target=targets[v];
    const auto gain=gains[v];
    long long correction=0;
    for (std::int64_t e=static_cast<std::int64_t>(offsets[v])+lane;e<offsets[v+1];e+=32) {
        const auto u=neighbors[e];
        if (u==v || targets[u]<0 || !(gains[u]>gain || (gains[u]==gain && u<v))) continue;
        const auto w=static_cast<long long>(weights[e]);
        correction+=w*((targets[u]==target)-(targets[u]==source)-(labels[u]==target)+(labels[u]==source));
    }
    for (int shift=16;shift;shift/=2) correction+=__shfl_down_sync(0xffffffffU,correction,shift);
    if (!lane) flags[v]=gain+correction>=0;
}

template <typename OffsetT,typename VertexT,typename WeightT>
__global__ void incremental_update_cache_and_cut(
    std::int64_t count,const VertexT* movers,const OffsetT* offsets,const VertexT* neighbors,
    const WeightT* weights,const VertexT* labels,const std::int32_t* targets,const std::int32_t* flags,
    int k,std::uint64_t* cache,std::int64_t* delta) {
    const int lane=threadIdx.x&31;
    const auto i=static_cast<std::int64_t>(blockIdx.x)*8+threadIdx.x/32;
    if (i>=count) return;
    const auto v=movers[i],source=labels[v];
    const auto target=targets[v];
    long long gain=0;
    for (std::int64_t e=static_cast<std::int64_t>(offsets[v])+lane;e<offsets[v+1];e+=32) {
        const auto u=neighbors[e];
        if (u==v) continue;
        const auto weight=static_cast<unsigned long long>(weights[e]);
        atomicAdd(reinterpret_cast<unsigned long long*>(cache+static_cast<std::int64_t>(u)*k+source),0ULL-weight);
        atomicAdd(reinterpret_cast<unsigned long long*>(cache+static_cast<std::int64_t>(u)*k+target),weight);
        // Each affected undirected edge contributes once, including edges
        // whose two endpoints move concurrently.
        if (!flags[u] || u>v) {
            const auto new_neighbor=flags[u] ? targets[u] : labels[u];
            gain+=static_cast<long long>(weight)*((source!=labels[u])-(target!=new_neighbor));
        }
    }
    for (int shift=16;shift;shift/=2) gain+=__shfl_down_sync(0xffffffffU,gain,shift);
    if (!lane) atomicAdd(reinterpret_cast<unsigned long long*>(delta),static_cast<unsigned long long>(gain));
}

template <typename VertexT,typename WeightT>
__global__ void incremental_commit(
    std::int64_t count,const VertexT* movers,VertexT* labels,const std::int32_t* targets,
    const WeightT* vertex_weights,unsigned long long* part_weights,std::int32_t* locks,bool lp) {
    const auto i=static_cast<std::int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=count) return;
    const auto v=movers[i],source=labels[v];
    const auto target=targets[v];
    const auto weight=static_cast<unsigned long long>(vertex_weights[v]);
    atomicAdd(part_weights+source,0ULL-weight);atomicAdd(part_weights+target,weight);
    labels[v]=static_cast<VertexT>(target);
    if (lp) locks[v]=1;
}

template <typename VertexT,typename WeightT>
__global__ void incremental_rebalance_proposals(
    std::int64_t n,const std::uint64_t* cache,const VertexT* labels,const WeightT* vertex_weights,
    int k,const unsigned long long* part_weights,std::uint64_t capacity,
    std::int32_t* targets,RebalanceProposal* proposals) {
    const auto v=static_cast<std::int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (v>=n) return;
    RebalanceProposal proposal{kInvalidTarget, 0.0};
    const auto source=labels[v];
    const auto weight=static_cast<std::uint64_t>(vertex_weights[v]);
    targets[v]=-1;
    if (part_weights[source]>capacity) {
        int best=-1;
        std::uint64_t best_conn=0,best_slack=0;
        for (int p=0;p<k;++p) {
            if (p==source || part_weights[p]>=capacity || weight>capacity-part_weights[p]) continue;
            const auto value=cache[v*k+p];
            const auto slack=capacity-part_weights[p];
            if (best<0 || value>best_conn || (value==best_conn && slack>best_slack)) {
                best=p;best_conn=value;best_slack=slack;
            }
        }
        // Spread vertices with no destination affinity across available slack,
        // instead of sending every isolated vertex to the same partition.
        if (best>=0 && best_conn==0) {
            std::uint64_t total_slack=0;
            for (int p=0;p<k;++p)
                if (p!=source && part_weights[p]<capacity && weight<=capacity-part_weights[p])
                    total_slack=UINT64_MAX-total_slack<capacity-part_weights[p] ? UINT64_MAX : total_slack+capacity-part_weights[p];
            auto ticket=static_cast<std::uint64_t>(refine_mix32(static_cast<std::uint32_t>(v)))%total_slack;
            for (int p=0;p<k;++p) {
                if (p==source || part_weights[p]>=capacity || weight>capacity-part_weights[p]) continue;
                const auto slack=capacity-part_weights[p];
                if (ticket<slack) { best=p;break; }
                ticket-=slack;
            }
        }
        if (best>=0) {
            const auto current=cache[v*k+source];
            const double gain=best_conn>=current ? static_cast<double>(best_conn-current) : -static_cast<double>(current-best_conn);
            proposal.source=source;
            proposal.gain_per_weight=gain/weight;
            targets[v]=best;
        }
    }
    proposals[v]=proposal;
}

struct IncrementalFlag {
    __host__ __device__ bool operator()(std::int32_t flag) const { return flag!=0; }
};

constexpr int kBalanceBuckets=50;
constexpr int kBalanceSections=16;
constexpr int kBalanceWidth=kBalanceBuckets*kBalanceSections;

__device__ int balance_gain_bucket(double gain) {
    if (gain>0) return 0;
    if (gain==0) return 1;
    return max(2,min(49,25+static_cast<int>(ceil(log(-gain)/log(1.5)))));
}

template <typename WeightT>
__global__ void balance_source_histogram(
    std::int64_t n,const RebalanceProposal* proposals,const WeightT* weights,
    std::int32_t* bucket_ids,std::uint64_t* ranks,unsigned long long* counts) {
    const auto v=static_cast<std::int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (v>=n) return;
    const auto proposal=proposals[v];
    bucket_ids[v]=-1;
    if (proposal.source==kInvalidTarget) return;
    const int bucket=proposal.source*kBalanceWidth+balance_gain_bucket(proposal.gain_per_weight)*kBalanceSections+
        refine_mix32(static_cast<std::uint32_t>(v))%kBalanceSections;
    bucket_ids[v]=bucket;
    ranks[v]=atomicAdd(counts+bucket,static_cast<unsigned long long>(weights[v]));
}

__global__ void balance_scan_buckets(
    const unsigned long long* counts,unsigned long long* offsets) {
    using Scan=cub::BlockScan<unsigned long long,256>;
    __shared__ typename Scan::TempStorage temp;
    unsigned long long values[4];
    for (int i=0;i<4;++i) {
        const int b=threadIdx.x*4+i;
        values[i]=b<kBalanceWidth ? counts[blockIdx.x*kBalanceWidth+b] : 0;
    }
    Scan(temp).ExclusiveSum(values,values);
    for (int i=0;i<4;++i) {
        const int b=threadIdx.x*4+i;
        if (b<kBalanceWidth) offsets[blockIdx.x*kBalanceWidth+b]=values[i];
    }
}

template <typename WeightT>
__global__ void balance_select_sources(
    std::int64_t n,std::int32_t* bucket_ids,std::uint64_t* ranks,
    const unsigned long long* offsets,const unsigned long long* part_weights,
    std::uint64_t capacity,const std::int32_t* targets,const WeightT* weights,
    unsigned long long* destination_counts) {
    const auto v=static_cast<std::int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (v>=n || bucket_ids[v]<0) return;
    const int b=bucket_ids[v],source=b/kBalanceWidth;
    if (offsets[b]+ranks[v]>=part_weights[source]-capacity) { bucket_ids[v]=-1;return; }
    const int destination=targets[v]*kBalanceWidth+b%kBalanceWidth;
    bucket_ids[v]=destination;
    ranks[v]=atomicAdd(destination_counts+destination,static_cast<unsigned long long>(weights[v]));
}

template <typename WeightT>
__global__ void balance_select_destinations(
    std::int64_t n,const std::int32_t* bucket_ids,const std::uint64_t* ranks,
    const unsigned long long* offsets,const unsigned long long* part_weights,
    std::uint64_t capacity,const WeightT* weights,std::int32_t* flags) {
    const auto v=static_cast<std::int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (v>=n) return;
    const int b=bucket_ids[v];
    flags[v]=0;
    if (b<0) return;
    const auto before=offsets[b]+ranks[v];
    const auto slack=capacity-part_weights[b/kBalanceWidth];
    flags[v]=before<=slack && static_cast<std::uint64_t>(weights[v])<=slack-before;
}
} // namespace

template <typename Types>
RefineStats run_incremental_refinement(
    RefineDeviceContext<Types>& context,const RefineOptions& options,int rounds) {
    const auto start=std::chrono::steady_clock::now();
    validate_refinement_gain_range(context.graph);
    auto& ws=context.workspace;
    auto& graph=context.graph;
    const auto n=context.n;
    const auto k=options.parts;
    ws.grow(ws.dense_affinity,static_cast<std::size_t>(n)*k);
    ws.grow(ws.incremental_targets,n);ws.grow(ws.incremental_gains,n);
    ws.grow(ws.incremental_locks,n);ws.grow(ws.incremental_flags,n);
    ws.grow(ws.incremental_movers,n);ws.grow(ws.incremental_delta,1);
    ws.grow(ws.balance_bucket_ids,n);ws.grow(ws.balance_bucket_ranks,n);
    ws.grow(ws.balance_bucket_counts,k*kBalanceWidth);
    ws.grow(ws.balance_bucket_offsets,k*kBalanceWidth);
    auto* cache=thrust::raw_pointer_cast(ws.dense_affinity.data());
    auto* targets=thrust::raw_pointer_cast(ws.incremental_targets.data());
    auto* gains=thrust::raw_pointer_cast(ws.incremental_gains.data());
    auto* locks=thrust::raw_pointer_cast(ws.incremental_locks.data());
    auto* flags=thrust::raw_pointer_cast(ws.incremental_flags.data());
    auto* labels=thrust::raw_pointer_cast(context.partition.data());
    auto* offsets=thrust::raw_pointer_cast(graph.offsets.data());
    auto* neighbors=thrust::raw_pointer_cast(graph.neighbors.data());
    auto* weights=thrust::raw_pointer_cast(graph.edge_weights.data());
    auto* vertex_weights=thrust::raw_pointer_cast(graph.vertex_weights.data());
    auto* part_weights=thrust::raw_pointer_cast(context.part_weights.data());
    auto* movers=thrust::raw_pointer_cast(ws.incremental_movers.data());
    thrust::fill(ws.incremental_locks.begin(),ws.incremental_locks.begin()+n,0);
    incremental_build_affinity<<<context.warp_blocks,256>>>(n,offsets,neighbors,weights,labels,k,cache);
    CUDA_CHECK(cudaGetLastError());
    RefineStats stats;
    stats.initial_cut=context.cut();
    auto current_cut=stats.initial_cut,best_cut=current_cut;
    const auto checkpoint=[&]() {
        thrust::copy(context.partition.begin(),context.partition.end(),ws.previous_partition.begin());
        thrust::copy(context.part_weights.begin(),context.part_weights.end(),ws.previous_part_weights.begin());
    };
    checkpoint();
    const auto maximum=[&]() { return *std::max_element(context.host_part_weights.begin(),context.host_part_weights.end()); };
    const auto all=thrust::make_counting_iterator<typename Types::VertexT>(0);
    const auto compact_movers=[&]() {
        const auto end=thrust::copy_if(thrust::device,all,all+n,ws.incremental_flags.begin(),ws.incremental_movers.begin(),IncrementalFlag{});
        return static_cast<std::int64_t>(end-ws.incremental_movers.begin());
    };
    const auto apply=[&](std::int64_t count,bool lp) {
        CUDA_CHECK(cudaMemset(thrust::raw_pointer_cast(ws.incremental_delta.data()),0,sizeof(std::int64_t)));
        incremental_update_cache_and_cut<<<static_cast<int>((count+7)/8),256>>>(count,movers,offsets,neighbors,weights,labels,targets,flags,k,cache,thrust::raw_pointer_cast(ws.incremental_delta.data()));
        CUDA_CHECK(cudaGetLastError());
        incremental_commit<<<static_cast<int>((count+255)/256),256>>>(count,movers,labels,targets,vertex_weights,part_weights,locks,lp);
        CUDA_CHECK(cudaGetLastError());
        std::int64_t delta=0;
        CUDA_CHECK(cudaMemcpy(&delta,thrust::raw_pointer_cast(ws.incremental_delta.data()),sizeof(delta),cudaMemcpyDeviceToHost));
        if (delta>=0) {
            if (static_cast<std::uint64_t>(delta)>current_cut) throw std::runtime_error("incremental cut underflow");
            current_cut-=static_cast<std::uint64_t>(delta);
        } else current_cut+=static_cast<std::uint64_t>(-delta);
        context.sync_part_weights();
        stats.accepted+=count;
        if (options.strict_verify) {
            if (context.cut()!=current_cut) throw std::runtime_error("incremental cut differs from CSR scan");
            detail::ScratchVector<std::uint64_t> check_cache(static_cast<std::size_t>(n)*k);
            incremental_build_affinity<<<context.warp_blocks,256>>>(n,offsets,neighbors,weights,labels,k,thrust::raw_pointer_cast(check_cache.data()));
            CUDA_CHECK(cudaGetLastError());
            if (!thrust::equal(check_cache.begin(),check_cache.end(),ws.dense_affinity.begin()))
                throw std::runtime_error("incremental affinity cache differs from CSR rebuild");
        }
    };
    const auto rebalance=[&]() {
        for (int iteration=0;iteration<2*k+4 && maximum()>context.capacity;++iteration) {
            ++stats.admission_iterations;
            incremental_rebalance_proposals<<<context.vertex_blocks,256>>>(n,cache,labels,vertex_weights,k,part_weights,context.capacity,targets,thrust::raw_pointer_cast(ws.rebalance_proposals.data()));
            CUDA_CHECK(cudaGetLastError());
            auto* ids=thrust::raw_pointer_cast(ws.balance_bucket_ids.data());
            auto* ranks=thrust::raw_pointer_cast(ws.balance_bucket_ranks.data());
            auto* counts=thrust::raw_pointer_cast(ws.balance_bucket_counts.data());
            auto* offsets=thrust::raw_pointer_cast(ws.balance_bucket_offsets.data());
            const auto bytes=k*kBalanceWidth*sizeof(unsigned long long);
            CUDA_CHECK(cudaMemset(counts,0,bytes));
            balance_source_histogram<<<context.vertex_blocks,256>>>(n,thrust::raw_pointer_cast(ws.rebalance_proposals.data()),vertex_weights,ids,ranks,counts);
            CUDA_CHECK(cudaGetLastError());
            balance_scan_buckets<<<k,256>>>(counts,offsets);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemset(counts,0,bytes));
            balance_select_sources<<<context.vertex_blocks,256>>>(n,ids,ranks,offsets,part_weights,context.capacity,targets,vertex_weights,counts);
            CUDA_CHECK(cudaGetLastError());
            balance_scan_buckets<<<k,256>>>(counts,offsets);
            CUDA_CHECK(cudaGetLastError());
            balance_select_destinations<<<context.vertex_blocks,256>>>(n,ids,ranks,offsets,part_weights,context.capacity,vertex_weights,flags);
            CUDA_CHECK(cudaGetLastError());
            const auto actual=compact_movers();
            if (!actual) return false;
            apply(actual,false);
        }
        return maximum()<=context.capacity;
    };
    using WeightT=typename Types::WeightT;
    const auto maximum_edge=thrust::reduce(graph.edge_weights.begin(),graph.edge_weights.end(),WeightT{0},thrust::maximum<WeightT>());
    const double filter=maximum_edge<=1 ? 0.25 : 0.75;
    int stale=0,empty=0;
    for (int phase=0;phase<rounds && stale<options.stale_rounds;++phase) {
        stats.rounds=phase+1;
        incremental_lp_proposals<<<context.vertex_blocks,256>>>(n,cache,labels,k,filter,locks,targets,gains);
        CUDA_CHECK(cudaGetLastError());
        incremental_priority_filter<<<context.warp_blocks,256>>>(n,offsets,neighbors,weights,labels,targets,gains,flags);
        CUDA_CHECK(cudaGetLastError());
        const auto count=compact_movers();
        stats.proposals+=count;
        if (!count) { if (++empty>=2) break;++stale;continue; }
        empty=0;
        apply(count,true);
        if (!rebalance()) { ++stats.admission_fallbacks;break; }
        ++stale;
        if (current_cut<best_cut) {
            if (static_cast<long double>(current_cut)<0.999L*best_cut) stale=0;
            best_cut=current_cut;checkpoint();
        }
    }
    thrust::copy(ws.previous_partition.begin(),ws.previous_partition.begin()+n,context.partition.begin());
    thrust::copy(ws.previous_part_weights.begin(),ws.previous_part_weights.begin()+k,context.part_weights.begin());
    context.sync_part_weights();
    stats.final_cut=context.cut();stats.final_max_part_weight=maximum();
    if (stats.final_cut!=best_cut || stats.final_cut>stats.initial_cut || maximum()>context.capacity)
        throw std::runtime_error("incremental refinement postcondition failed");
    stats.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    return stats;
}
