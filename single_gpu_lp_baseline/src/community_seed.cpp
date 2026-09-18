// Flat structural-group LP reference. All edge scans use the original CSR.
#include "graph.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

int main(int argc,char** argv)try{
    if(argc!=10)throw std::runtime_error("usage: community_seed ptr idx initial.parts output.parts k block_limit community_rounds block_rounds seed");
    CSRGraph g;g.load(argv[1],argv[2]);const auto& ptr=g.offsets();const auto& idx=g.neighbors();
    const int n=g.vertices(),k=std::stoi(argv[5]),limit=std::stoi(argv[6]);
    const int rounds=std::stoi(argv[7]),block_rounds=std::stoi(argv[8]);std::mt19937 rng(std::stoul(argv[9]));
    if(k<2||k>32||limit<1||rounds<1||block_rounds<1)throw std::runtime_error("invalid search parameters");
    const long long cap=std::floor((long double)n/k*1.10L);
    std::vector<int> initial(n),cluster(n),size(n,1),order(n),counts(n),stamp(n,-1);
    std::ifstream in(argv[3],std::ios::binary|std::ios::ate);if(!in||in.tellg()!=std::streamoff(n*sizeof(int)))throw std::runtime_error("bad initial labels");
    in.seekg(0);in.read((char*)initial.data(),n*sizeof(int));for(int p:initial)if(p<0||p>=k)throw std::runtime_error("bad label");
    std::iota(cluster.begin(),cluster.end(),0);std::iota(order.begin(),order.end(),0);
    std::shuffle(order.begin(),order.end(),rng);
    std::stable_sort(order.begin(),order.end(),[&](int a,int b){return ptr[a+1]-ptr[a]<ptr[b+1]-ptr[b];});
    auto start=std::chrono::steady_clock::now();int epoch=0;std::vector<int> touched;
    for(int r=0;r<rounds;++r){int moved=0;
        if(r>0)std::shuffle(order.begin(),order.end(),rng);
        for(int v:order){++epoch;touched.clear();int own=cluster[v];
            for(auto e=ptr[v];e<ptr[v+1];++e){int c=cluster[idx[e]];if(stamp[c]!=epoch){stamp[c]=epoch;counts[c]=0;touched.push_back(c);}++counts[c];}
            int own_count=stamp[own]==epoch?counts[own]:0,best=own,best_count=own_count;
            for(int c:touched)if(c!=own&&size[c]<limit&&(counts[c]>best_count||(counts[c]==best_count&&best!=own&&size[c]<size[best]))){best=c;best_count=counts[c];}
            if(best!=own){--size[own];++size[best];cluster[v]=best;++moved;}
        }
        std::cout<<"community_round="<<r<<" moved="<<moved<<std::endl;if(!moved)break;
    }
    std::vector<int> head(n,-1),next(n,-1),blocks,part(n,-1),best_part(n,-1);
    for(int v=0;v<n;++v){next[v]=head[cluster[v]];head[cluster[v]]=v;}
    std::vector<long long> load(k);
    for(int c=0;c<n;++c)if(size[c]){blocks.push_back(c);std::vector<int> votes(k);
        for(int v=head[c];v!=-1;v=next[v])++votes[initial[v]];
        part[c]=std::max_element(votes.begin(),votes.end())-votes.begin();load[part[c]]+=size[c];
    }
    auto cut=[&](){long long result=0;for(int v=0;v<n;++v)for(auto e=ptr[v];e<ptr[v+1];++e)result+=part[cluster[v]]!=part[cluster[idx[e]]];return result/2;};
    long long best_cut=std::numeric_limits<long long>::max();
    std::cout<<"blocks="<<blocks.size()<<" limit="<<limit<<" projected_cut="<<cut()<<std::endl;
    for(int r=0;r<block_rounds;++r){std::shuffle(blocks.begin(),blocks.end(),rng);int moved=0;
        for(int c:blocks){int a=part[c];long long edges[32]={};
            for(int v=head[c];v!=-1;v=next[v])for(auto e=ptr[v];e<ptr[v+1];++e){int other=cluster[idx[e]];if(other!=c)++edges[part[other]];}
            int best=a;long long gain=0;bool overloaded=load[a]>cap;
            for(int b=0;b<k;++b)if(b!=a&&load[b]+size[c]<=cap){long long candidate=edges[b]-edges[a];
                if((overloaded&&best==a)||candidate>gain){best=b;gain=candidate;}}
            if(best!=a){load[a]-=size[c];load[best]+=size[c];part[c]=best;++moved;}
        }
        auto current=cut();bool feasible=*std::max_element(load.begin(),load.end())<=cap;
        std::cout<<"block_round="<<r<<" moved="<<moved<<" cut="<<current<<" ratio="<<2.0*current/g.edges()<<" feasible="<<feasible<<std::endl;
        if(feasible&&current<best_cut){best_cut=current;best_part=part;}
        if(!moved)break;
    }
    if(best_cut==std::numeric_limits<long long>::max())throw std::runtime_error("no feasible block labeling");
    for(int v=0;v<n;++v)initial[v]=best_part[cluster[v]];
    std::ofstream out(argv[4],std::ios::binary);out.write((char*)initial.data(),n*sizeof(int));if(!out)throw std::runtime_error("cannot write output");
    std::cout<<"final_cut="<<best_cut<<" cut_ratio="<<2.0*best_cut/g.edges()<<" community_seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<std::endl;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
