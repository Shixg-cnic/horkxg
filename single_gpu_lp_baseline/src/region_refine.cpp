// Original-graph regional min-cut proposals plus exact-gain capacity repair.
#include "graph.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <tuple>

struct Flow {
    struct Edge{int to,rev,cap;};
    std::vector<std::vector<Edge>> edges;std::vector<int> level,pos;
    explicit Flow(int n):edges(n),level(n),pos(n){}
    void add(int a,int b,int cap,int reverse=0){int ai=edges[a].size(),bi=edges[b].size();edges[a].push_back({b,bi,cap});edges[b].push_back({a,ai,reverse});}
    int dfs(int v,int t,int f){if(v==t)return f;for(int& i=pos[v];i<(int)edges[v].size();++i){auto& e=edges[v][i];if(e.cap&&level[e.to]==level[v]+1){int d=dfs(e.to,t,std::min(f,e.cap));if(d){e.cap-=d;edges[e.to][e.rev].cap+=d;return d;}}}return 0;}
    void solve(int s,int t){while(true){std::fill(level.begin(),level.end(),-1);std::queue<int>q;q.push(s);level[s]=0;while(!q.empty()){int v=q.front();q.pop();for(auto e:edges[v])if(e.cap&&level[e.to]<0){level[e.to]=level[v]+1;q.push(e.to);}}if(level[t]<0)break;std::fill(pos.begin(),pos.end(),0);while(dfs(s,t,1000000000)){} }}
    std::vector<unsigned char> balanced_side(int s,int t,int low_b,int high_b,const std::vector<unsigned char>& old_b){
        std::vector<unsigned char> result(s);int nb=0;for(int v=0;v<s;++v){result[v]=level[v]<0;nb+=result[v];}
        if(nb<=high_b)return result;
        int n=edges.size();std::vector<unsigned char> visited(n);std::vector<int> order,component(n,-1);
        for(int root=0;root<n;++root)if(!visited[root]){
            std::vector<std::pair<int,int>> stack{{root,0}};visited[root]=1;
            while(!stack.empty()){auto& frame=stack.back();int v=frame.first;bool added=false;
                while(frame.second<(int)edges[v].size()){auto e=edges[v][frame.second++];if(e.cap&&!visited[e.to]){visited[e.to]=1;stack.emplace_back(e.to,0);added=true;break;}}
                if(!added){order.push_back(v);stack.pop_back();}
            }
        }
        int components=0;
        for(auto it=order.rbegin();it!=order.rend();++it)if(component[*it]<0){std::vector<int> stack{*it};component[*it]=components;
            while(!stack.empty()){int v=stack.back();stack.pop_back();for(auto e:edges[v])if(edges[e.to][e.rev].cap&&component[e.to]<0){component[e.to]=components;stack.push_back(e.to);}}++components;
        }
        std::vector<std::vector<int>> dag(components),reverse(components);std::vector<int> weight(components),preference(components);
        std::vector<unsigned char> side_a(components),side_b(components);
        for(int v=0;v<n;++v){int c=component[v];if(v<s){++weight[c];preference[c]+=!old_b[v];}if(level[v]>=0)side_a[c]=1;
            for(auto e:edges[v])if(e.cap&&c!=component[e.to])dag[c].push_back(component[e.to]);}
        for(int c=0;c<components;++c){auto& d=dag[c];std::sort(d.begin(),d.end());d.erase(std::unique(d.begin(),d.end()),d.end());for(int other:d)reverse[other].push_back(c);}
        std::vector<int> stack{component[t]};side_b[component[t]]=1;
        while(!stack.empty()){int c=stack.back();stack.pop_back();for(int other:reverse[c])if(!side_b[other]){side_b[other]=1;stack.push_back(other);}}
        std::vector<int> roots;for(int c=0;c<components;++c)if(weight[c]&&!side_a[c]&&!side_b[c])roots.push_back(c);
        std::sort(roots.begin(),roots.end(),[&](int x,int y){long long left=(long long)preference[x]*weight[y],right=(long long)preference[y]*weight[x];return left!=right?left>right:(weight[x]!=weight[y]?weight[x]<weight[y]:x<y);});
        std::vector<int> stamp(components,-1);int epoch=0;
        for(int root:roots){if(nb<=high_b)break;if(side_a[root])continue;++epoch;std::vector<int> closure{root};stamp[root]=epoch;int added=0;
            for(size_t q=0;q<closure.size();++q){int c=closure[q];added+=weight[c];if(added>nb-low_b)break;for(int other:dag[c])if(!side_a[other]&&stamp[other]!=epoch){stamp[other]=epoch;closure.push_back(other);}}
            if(added<=nb-low_b){for(int c:closure)side_a[c]=1;nb-=added;}
        }
        for(int v=0;v<n;++v)if(side_a[component[v]])for(auto e:edges[v])if(e.cap&&!side_a[component[e.to]])throw std::runtime_error("residual closure violated");
        for(int v=0;v<s;++v)result[v]=!side_a[component[v]];
        return result;
    }
};
struct Move{int gain,v,version;bool operator<(const Move& b)const{return std::tie(gain,v)<std::tie(b.gain,b.v);}};

int main(int argc,char** argv)try{
    if(argc!=9)throw std::runtime_error("usage: region_refine ptr idx initial.parts output.parts k region_size passes seed");
    CSRGraph graph;graph.load(argv[1],argv[2]);const auto& off=graph.offsets();const auto& adj=graph.neighbors();
    int n=graph.vertices(),k=std::stoi(argv[5]),region_size=std::stoi(argv[6]),passes=std::stoi(argv[7]);
    if(k<2||k>32||region_size<2||region_size>65536||passes<1)throw std::runtime_error("invalid configuration");
    const long double ratio=std::getenv("REGION_MAX_RATIO")?std::stold(std::getenv("REGION_MAX_RATIO")):1.10L;
    if(ratio<1.0L||ratio>1.5L)throw std::runtime_error("REGION_MAX_RATIO outside 1..1.5");
    long long cap=std::floor((long double)n/k*ratio);
    const int lagrange_steps=std::getenv("REGION_LAGRANGE_STEPS")?std::stoi(std::getenv("REGION_LAGRANGE_STEPS")):0;
    const double time_budget=std::getenv("REGION_SECONDS")?std::stod(std::getenv("REGION_SECONDS")):0;
    const bool residual_balance=std::getenv("REGION_RESIDUAL_BALANCE")&&std::string(std::getenv("REGION_RESIDUAL_BALANCE"))!="0";
    std::vector<int> label(n),count((size_t)n*k),order(n),seen(n,-1),id(n),version(n);std::vector<long long> load(k);
    std::ifstream in(argv[3],std::ios::binary|std::ios::ate);if(!in||in.tellg()!=std::streamoff(n*sizeof(int)))throw std::runtime_error("bad initial size");in.seekg(0);in.read((char*)label.data(),n*sizeof(int));
    for(int p:label){if(p<0||p>=k)throw std::runtime_error("invalid label");++load[p];}for(auto l:load)if(l>cap)throw std::runtime_error("initial labels infeasible");
    auto exact_cut=[&](){long long cut=0;for(int v=0;v<n;++v)for(auto e=off[v];e<off[v+1];++e)cut+=label[v]!=label[adj[e]];return cut/2;};
    long long cut=exact_cut();auto start=std::chrono::steady_clock::now();
    for(int v=0;v<n;++v)for(auto e=off[v];e<off[v+1];++e)++count[(size_t)v*k+label[adj[e]]];
    std::iota(order.begin(),order.end(),0);std::mt19937 rng(std::stoul(argv[8]));int epoch=0;
    std::cout<<"initial_cut="<<cut<<" ratio="<<2.0*cut/graph.edges()<<" search_max_ratio="<<double(ratio)<<std::endl;
    for(int pass=0;pass<passes;++pass){std::shuffle(order.begin(),order.end(),rng);std::vector<unsigned char> visited(n,0);long long saved=0;int attempts=0,accepted=0;
        for(int seed:order){if(time_budget>0&&attempts%128==0&&std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()>time_budget)break;if(visited[seed])continue;int a=label[seed],b=-1;
            for(int p=0;p<k;++p)if(p!=a&&count[(size_t)seed*k+p]>0&&(b<0||count[(size_t)seed*k+p]>count[(size_t)seed*k+b]))b=p;
            if(b<0)continue;++epoch;++attempts;std::vector<int> region;region.reserve(region_size);
            auto add=[&](int v){if(!visited[v]&&(label[v]==a||label[v]==b)&&(int)region.size()<region_size){visited[v]=1;seen[v]=epoch;id[v]=region.size();region.push_back(v);}};
            add(seed);for(size_t q=0;q<region.size()&&(int)region.size()<region_size;++q){int v=region[q];for(auto e=off[v];e<off[v+1]&&(int)region.size()<region_size;++e)add(adj[e]);}
            int s=region.size(),t=s+1;Flow flow(t+1);
            for(int v:region){int ca=0,cb=0;for(auto e=off[v];e<off[v+1];++e){int u=adj[e];if(seen[u]==epoch){if(v<u)flow.add(id[v],id[u],1,1);}else{ca+=label[u]==a;cb+=label[u]==b;}}
                if(ca)flow.add(s,id[v],ca);if(cb)flow.add(id[v],t,cb);
            }
            flow.solve(s,t);
            std::vector<unsigned char> proposal(s);int old_b=0,new_b=0;
            for(int v:region){old_b+=label[v]==b;proposal[id[v]]=flow.level[id[v]]<0;new_b+=proposal[id[v]];}
            const int low=std::max<long long>(0,old_b+load[a]-cap),high=std::min<long long>(s,old_b+cap-load[b]);
            std::vector<unsigned char> old_side(s);for(int v:region)old_side[id[v]]=label[v]==b;
            if(residual_balance){proposal=flow.balanced_side(s,t,low,high,old_side);new_b=std::accumulate(proposal.begin(),proposal.end(),0);}
            auto excess=[&](int nb){return std::max(0,low-nb)+std::max(0,nb-high);};
            auto cost=[&](const std::vector<unsigned char>& p){long long c=0;for(int v:region)for(auto e=off[v];e<off[v+1];++e){int u=adj[e];if(seen[u]==epoch){if(v<u)c+=p[id[v]]!=p[id[u]];}else c+=(p[id[v]]?b:a)!=label[u];}return c;};
            if(lagrange_steps>0&&excess(new_b)>0){
                int best_excess=excess(new_b);auto best_cost=cost(proposal);int target=new_b>high?high:low,left=-256,right=256;
                for(int step=0;step<lagrange_steps&&left<=right;++step){int penalty=(left+right)/2;Flow trial(t+1);
                    for(int v:region){int ca=0,cb=0;for(auto e=off[v];e<off[v+1];++e){int u=adj[e];if(seen[u]==epoch){if(v<u)trial.add(id[v],id[u],64,64);}else{ca+=label[u]==a;cb+=label[u]==b;}}
                        trial.add(s,id[v],ca*64+std::max(0,penalty));trial.add(id[v],t,cb*64+std::max(0,-penalty));}
                    trial.solve(s,t);std::vector<unsigned char> p(s);int nb=0;for(int i=0;i<s;++i){p[i]=trial.level[i]<0;nb+=p[i];}
                    if(residual_balance){p=trial.balanced_side(s,t,low,high,old_side);nb=std::accumulate(p.begin(),p.end(),0);}
                    int ex=excess(nb);auto c=cost(p);if(ex<best_excess||(ex==best_excess&&c<best_cost)){best_excess=ex;best_cost=c;proposal=std::move(p);}
                    if(nb>target)left=penalty+1;else right=penalty-1;
                }
            }
            std::vector<std::tuple<int,int,int>> history;long long gain=0;
            auto move=[&](int v,int target){int source=label[v];if(source==target)return;gain+=count[(size_t)v*k+target]-count[(size_t)v*k+source];history.emplace_back(v,source,target);label[v]=target;--load[source];++load[target];
                for(auto e=off[v];e<off[v+1];++e){int u=adj[e];--count[(size_t)u*k+source];++count[(size_t)u*k+target];++version[u];}};
            for(int v:region)move(v,proposal[id[v]]?b:a);
            int source=load[a]>cap?a:(load[b]>cap?b:-1);
            if(source>=0){int target=source==a?b:a;std::priority_queue<Move> queue;
                auto push=[&](int v){if(seen[v]==epoch&&label[v]==source)queue.push({count[(size_t)v*k+target]-count[(size_t)v*k+source],v,version[v]});};
                for(int v:region)push(v);
                while(load[source]>cap&&!queue.empty()){auto e=queue.top();queue.pop();if(label[e.v]!=source||e.version!=version[e.v])continue;move(e.v,target);for(auto i=off[e.v];i<off[e.v+1];++i)push(adj[i]);}
            }
            // During loose exploration retain labels as seeds for later repair.
            if(gain>0&&load[a]<=cap&&load[b]<=cap&&(ratio<=1.10L||(load[a]>0&&load[b]>0))){cut-=gain;saved+=gain;++accepted;}
            else{for(auto it=history.rbegin();it!=history.rend();++it){auto [v,source,target]=*it;label[v]=source;++load[source];--load[target];for(auto e=off[v];e<off[v+1];++e){int u=adj[e];++count[(size_t)u*k+source];--count[(size_t)u*k+target];++version[u];}}}
        }
        if(exact_cut()!=cut)throw std::runtime_error("full cut verification failed");for(auto l:load)if(l>cap)throw std::runtime_error("capacity violation");
        std::cout<<"pass="<<pass<<" attempts="<<attempts<<" accepted="<<accepted<<" gain="<<saved<<" cut="<<cut<<" ratio="<<2.0*cut/graph.edges()<<std::endl;
        {std::ofstream checkpoint(argv[4],std::ios::binary);checkpoint.write((char*)label.data(),n*sizeof(int));if(!checkpoint)throw std::runtime_error("checkpoint write failed");}
        if(time_budget>0&&std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()>time_budget)break;
        if(!saved)break;
    }
    std::ofstream out(argv[4],std::ios::binary);out.write((char*)label.data(),n*sizeof(int));if(!out)throw std::runtime_error("write failed");
    std::cout<<"final_cut="<<cut<<" cut_ratio="<<2.0*cut/graph.edges()<<" region_seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<" vertex_imb="<<double(*std::max_element(load.begin(),load.end()))*k/n<<std::endl;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
