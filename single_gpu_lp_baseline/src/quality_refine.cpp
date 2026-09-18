// Single-level reference search. No contraction; exact gains on original CSR.
#include "graph.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <queue>
#include <random>
#include <stdexcept>
#include <tuple>

struct Entry {
    int gain, vertex, version;
    unsigned tie;
    bool operator<(const Entry& b) const {
        return std::tie(gain,tie) < std::tie(b.gain,b.tie);
    }
};
int main(int argc, char** argv) try {
    if(argc != 11 && argc != 12) throw std::runtime_error("usage: quality_refine indptr indices input.parts output.parts k ratio passes patience seed swaps [group_size]");
    CSRGraph graph; graph.load(argv[1],argv[2]);
    const auto& off=graph.offsets(); const auto& adj=graph.neighbors();
    const int n=graph.vertices(), k=std::stoi(argv[5]);
    if(k<2 || k>32) throw std::runtime_error("k outside 2..32");
    const long long cap=std::floor((long double)n/k*std::stold(argv[6]));
    const int passes=std::stoi(argv[7]), patience=std::stoi(argv[8]);
    std::mt19937 rng(std::stoul(argv[9]));
    // argv[10] controls whether capacity-neutral swaps are enabled.
    const bool swaps=std::stoi(argv[10]);
    const int group_size=argc==12?std::stoi(argv[11]):0;
    const int loss_limit=std::getenv("GROUP_LOSS_LIMIT")?std::stoi(std::getenv("GROUP_LOSS_LIMIT")):256;
    const bool group_exchange=std::getenv("GROUP_EXCHANGE")&&std::string(std::getenv("GROUP_EXCHANGE"))!="0";
    const bool group_redirect=std::getenv("GROUP_REDIRECT")&&std::string(std::getenv("GROUP_REDIRECT"))!="0";
    std::vector<int> label(n), count((size_t)n*k), version(n);
    std::vector<long long> load(k);
    std::ifstream input(argv[3],std::ios::binary|std::ios::ate);
    if(!input || input.tellg()!=std::streamoff(n*sizeof(int))) throw std::runtime_error("invalid input partition size");
    input.seekg(0); input.read((char*)label.data(),n*sizeof(int));
    for(int p:label) {if(p<0||p>=k) throw std::runtime_error("invalid label"); ++load[p];}
    for(auto l:load) if(l>cap) throw std::runtime_error("input exceeds capacity");
    auto exact_cut=[&](){long long c=0; for(int v=0;v<n;++v) for(auto e=off[v];e<off[v+1];++e) c+=label[v]!=label[adj[e]]; return c/2;};
    long long cut=exact_cut();
    auto start=std::chrono::steady_clock::now();
    for(int v=0;v<n;++v) for(auto e=off[v];e<off[v+1];++e) ++count[(size_t)v*k+label[adj[e]]];
    std::cout<<"initial_cut="<<cut<<" ratio="<<2.0*cut/graph.edges()<<std::endl;
    for(int pass=0;pass<passes;++pass) {
        if(group_size>0) {
            std::vector<int> order(n), stamp(n,-1), bonus(n), selected(n,-1);
            std::vector<int> donor_stamp(n,-1);
            std::vector<int> donor_version(n,0);
            std::vector<std::vector<int>> pools(k*k);
            if(group_exchange) {
                for(int v=0;v<n;++v)for(int p=0;p<k;++p)
                    if(p!=label[v]&&count[(size_t)v*k+p]>0)pools[label[v]*k+p].push_back(v);
                for(int a=0;a<k;++a)for(int b=0;b<k;++b)if(a!=b){
                    auto& pool=pools[a*k+b];
                    auto rank=[&](int x,int y){int gx=count[(size_t)x*k+b]-count[(size_t)x*k+a],gy=count[(size_t)y*k+b]-count[(size_t)y*k+a];return gx!=gy?gx>gy:x<y;};
                    if(pool.size()>512){std::nth_element(pool.begin(),pool.begin()+512,pool.end(),rank);pool.resize(512);}
                    std::sort(pool.begin(),pool.end(),rank);
                }
            }
            std::vector<unsigned char> tried(n,0);
            for(int v=0;v<n;++v)order[v]=v;
            std::shuffle(order.begin(),order.end(),rng);
            long long group_gain=0;int accepted=0,attempt=0,exchange_accepted=0,exchange_rejected=0;
            for(int seed:order) {
                if(tried[seed])continue;
                int a=label[seed],b=-1;auto c=&count[(size_t)seed*k];
                for(int p=0;p<k;++p)if(p!=a&&(group_exchange||load[p]<cap)&&c[p]>0&&(b<0||c[p]>c[b]))b=p;
                if(b<0)continue;
                ++attempt;
                std::priority_queue<Entry> frontier;
                frontier.push({c[b]-c[a],seed,0,0});stamp[seed]=attempt;bonus[seed]=0;
                std::vector<int> group;long long gain=0,best=0;size_t prefix=0;
                const int limit=group_exchange?group_size:std::min<long long>(group_size,cap-load[b]);
                while(!frontier.empty()&&(int)group.size()<limit) {
                    auto e=frontier.top();frontier.pop();int v=e.vertex;
                    if(selected[v]==attempt||e.version!=bonus[v])continue;
                    selected[v]=attempt;tried[v]=1;group.push_back(v);gain+=e.gain;
                    if(gain>best){best=gain;prefix=group.size();}
                    // Bound wasted exploration of high-loss regions.
                    if(gain < -loss_limit)break;
                    for(auto i=off[v];i<off[v+1];++i){int u=adj[i];if(label[u]!=a||selected[u]==attempt)continue;
                        if(stamp[u]!=attempt){stamp[u]=attempt;bonus[u]=0;}bonus[u]+=2;
                        frontier.push({count[(size_t)u*k+b]-count[(size_t)u*k+a]+bonus[u],u,bonus[u],0});
                    }
                }
                if(best>0) {
                    std::vector<std::tuple<int,int,int>> transaction;
                    long long joint_gain=0;
                    auto apply=[&](int v,int target){int source=label[v];
                        joint_gain+=count[(size_t)v*k+target]-count[(size_t)v*k+source];
                        transaction.emplace_back(v,source,target);label[v]=target;--load[source];++load[target];
                        for(auto e=off[v];e<off[v+1];++e){int u=adj[e];--count[(size_t)u*k+source];++count[(size_t)u*k+target];}
                    };
                    for(size_t j=0;j<prefix;++j)apply(group[j],b);
                    const bool needs_exchange=load[b]>cap;
                    if(needs_exchange) {
                        std::vector<int> donors;
                        auto add=[&](int v){if(label[v]==b&&selected[v]!=attempt&&donor_stamp[v]!=attempt&&donors.size()<2048){donor_stamp[v]=attempt;donors.push_back(v);}};
                        if(group_redirect){for(int p=0;p<k;++p)if(p!=b&&load[p]<cap)for(int v:pools[b*k+p])add(v);}
                        else for(int v:pools[b*k+a])add(v);
                        for(size_t j=0;j<prefix;++j)for(auto e=off[group[j]];e<off[group[j]+1];++e)add(adj[e]);
                        if(group_redirect) {
                            std::priority_queue<Entry> queue;
                            auto target_for=[&](int v){int target=-1;for(int p=0;p<k;++p)if(p!=b&&load[p]<cap&&(target<0||count[(size_t)v*k+p]>count[(size_t)v*k+target]))target=p;return target;};
                            auto enqueue=[&](int v){if(label[v]!=b||donor_stamp[v]!=attempt)return;int target=target_for(v);if(target>=0)queue.push({count[(size_t)v*k+target]-count[(size_t)v*k+b],v,++donor_version[v],static_cast<unsigned>(v)});};
                            for(int v:donors)enqueue(v);
                            while(load[b]>cap&&!queue.empty()) {
                                auto entry=queue.top();queue.pop();int v=entry.vertex;
                                if(label[v]!=b||entry.version!=donor_version[v])continue;
                                int target=target_for(v);if(target<0)break;
                                int actual=count[(size_t)v*k+target]-count[(size_t)v*k+b];
                                if(actual!=entry.gain){enqueue(v);continue;}
                                apply(v,target);
                                for(auto e=off[v];e<off[v+1];++e)enqueue(adj[e]);
                            }
                        } else while(load[b]>cap) {
                            int chosen=-1,top=-1000000000;
                            for(int v:donors)if(label[v]==b){int g=count[(size_t)v*k+a]-count[(size_t)v*k+b];if(g>top){chosen=v;top=g;}}
                            if(chosen<0)break;
                            apply(chosen,a);
                        }
                    }
                    if(joint_gain>0 && load[b]<=cap && load[a]<=cap) {
                        cut-=joint_gain;group_gain+=joint_gain;++accepted;exchange_accepted+=needs_exchange;
                    } else {
                        for(auto it=transaction.rbegin();it!=transaction.rend();++it){auto [v,source,target]=*it;label[v]=source;++load[source];--load[target];
                            for(auto e=off[v];e<off[v+1];++e){int u=adj[e];++count[(size_t)u*k+source];--count[(size_t)u*k+target];}}
                        exchange_rejected+=needs_exchange;
                    }
                }
            }
            if(exact_cut()!=cut)throw std::runtime_error("group cut mismatch");
            std::cout<<"group_pass="<<pass<<" gain="<<group_gain<<" accepted="<<accepted<<" exchange_accepted="<<exchange_accepted<<" exchange_rejected="<<exchange_rejected<<" cut="<<cut<<" ratio="<<2.0*cut/graph.edges()<<std::endl;
        }
        std::vector<std::priority_queue<Entry>> heap(k*k);
        std::vector<unsigned char> locked(n,0);
        std::vector<std::tuple<int,int,int>> history;
        auto push=[&](int v){ if(locked[v])return; auto c=&count[(size_t)v*k]; int a=label[v];
            for(int b=0;b<k;++b) if(b!=a && c[b]>0) heap[a*k+b].push({c[b]-c[a],v,version[v],static_cast<unsigned>(rng())}); };
        for(int v=0;v<n;++v)push(v);
        auto peek=[&](int a,int b)->Entry {auto& h=heap[a*k+b];
            while(!h.empty()&&(locked[h.top().vertex]||version[h.top().vertex]!=h.top().version||label[h.top().vertex]!=a))h.pop();
            return h.empty()?Entry{-1000000000,-1,0,0}:h.top();};
        auto move=[&](int v,int b){int a=label[v]; history.emplace_back(v,a,b); locked[v]=1; label[v]=b; --load[a];++load[b];
            for(auto e=off[v];e<off[v+1];++e){int u=adj[e]; --count[(size_t)u*k+a];++count[(size_t)u*k+b];++version[u];push(u);} };
        long long gain=0,best=0; size_t best_pos=0; int stale=0, exchanges=0;
        while(stale<patience) {
            Entry candidate{-1000000000,-1,0,0};int target=-1,partner=-1;
            for(int a=0;a<k;++a)for(int b=0;b<k;++b) if(a!=b && load[b]<cap) {
                auto e=peek(a,b); if(candidate<e){candidate=e;target=b;partner=-1;}
            }
            if(swaps)for(int a=0;a<k;++a)for(int b=a+1;b<k;++b) {
                auto x=peek(a,b),y=peek(b,a); if(x.vertex<0||y.vertex<0)continue;
                int correction=0;for(auto e=off[x.vertex];e<off[x.vertex+1];++e)correction+=adj[e]==y.vertex;
                int g=x.gain+y.gain-2*correction;
                if(g>candidate.gain){candidate=x;candidate.gain=g;target=b;partner=y.vertex;}
            }
            if(candidate.vertex<0)break;
            int source=label[candidate.vertex];
            move(candidate.vertex,target); if(partner>=0){move(partner,source);++exchanges;}
            gain+=candidate.gain; ++stale;
            if(gain>best){best=gain;best_pos=history.size();stale=0;}
        }
        for(size_t i=history.size();i>best_pos;--i){auto [v,a,b]=history[i-1];label[v]=a;--load[b];++load[a];
            for(auto e=off[v];e<off[v+1];++e){int u=adj[e];--count[(size_t)u*k+b];++count[(size_t)u*k+a];}}
        cut-=best;
        if(exact_cut()!=cut)throw std::runtime_error("incremental cut mismatch");
        for(auto l:load)if(l>cap)throw std::runtime_error("capacity violation");
        std::cout<<"pass="<<pass<<" cut="<<cut<<" ratio="<<2.0*cut/graph.edges()<<" gain="<<best<<" kept_moves="<<best_pos<<" attempted="<<history.size()<<" swaps="<<exchanges<<std::endl;
        if(best==0 && !group_size)break;
    }
    auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::ofstream output(argv[4],std::ios::binary);output.write((char*)label.data(),n*sizeof(int));if(!output)throw std::runtime_error("output write failed");
    std::cout<<"final_cut="<<cut<<" cut_ratio="<<2.0*cut/graph.edges()<<" refine_seconds="<<seconds<<" vertex_imb="<<double(*std::max_element(load.begin(),load.end()))*k/n<<std::endl;
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
