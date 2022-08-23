#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <map>
#include <math.h>
#include <utility>
#include <cassert>
using namespace std;

/* 
    One counter each for each (train,class) pair
    Example: If Train 0 has 3 classes, train 1 has 2 classes,
    counts[0]counts[1]counts[2]   --> train 0, each class
    counts[3]counts[4]          --> train 1, each class
    Prefix sum to index into this
*/
__device__ volatile unsigned counts[2500000];	
__device__ volatile int reqs[40000];		
__device__ volatile int huge_arr_vol[25*50*100000];
__device__ volatile unsigned d_counter1;            //A global counter for the device

// Store Train Data
struct TrainInfo{
    int t_no;
    int num_classes;
    int *class_cap;
    int src;
    int dst;
};

// Set counter to 0
__global__ void initC(){
	d_counter1 = 0;
}
//Set all counts within len in counts array to 0
__global__ void setAllZeroes(int len){
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(tid < len){
        counts[tid] = 0;
    }
}
//Set all values in given array to true
__global__ void setAllTrue(bool *a,int len){
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(tid < len){
        a[tid] = true;
    }
}
__global__ void setReqs(int *a,int num_reqs){
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(tid < num_reqs*8){
        reqs[tid] = a[tid];
    }
}
__global__ void setHugeArr(int *a,int l){
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(tid < l){
        huge_arr_vol[tid] = a[tid];
    }
}
/*
Launch with <<< number of requests, maximum number of stations >>>
start_ind1  --> index into counts[], using train_number
start_ind2  --> index into huge_arr[], using train_number
huge_arr    --> One element each for each station for each class for each train holding capacity
Ex: Train 0, 2 classes 5,6 capacities, station 46-49 ==> 3 stations, 
    Train 1, 3 classes 2,3,4 capacities, stations 35-37 ==> 2 stations
    huge_arr ==> 5,5,5,6,6,6,2,2,3,3,4,4
reqs        --> Requests,each request has 8 integers associated
reqs[i*8+0] --> request number
reqs[i*8+1] --> train num
reqs[i*8+2] --> class num
reqs[i*8+3] --> smaller one of src,dst - smaller one of train's src,dst -> Rescale it to start at 0
reqs[i*8+4] --> larger one of src,dst -  smaller one of train's src,dst
reqs[i*8+5] --> num of seats to book
reqs[i*8+6] --> id for given (train_num, class_num) pair. Ids to maintain sequentiality for reqs with same (train_num, class_num) pair
reqs[i*8+7] --> larger of train's src,dst - smaller of train's src,dst --> number of stations of actual train
status      --> Store true if booking done, else false
*/
__global__ void BookIt(int *start_ind1,int *start_ind2, bool *status){
	__shared__ unsigned int cc;
    cc = 0;                         //Count number of invalid seats for req i.e block
    int rid_base = blockIdx.x*8;
    int tid = threadIdx.x;
    int tnum = reqs[rid_base+1];
    int cnum = reqs[rid_base+2];
    int src = reqs[rid_base+3];
    int dst = reqs[rid_base+4];
    int seats = reqs[rid_base+5];
    int src_dst = reqs[rid_base+7];
    int i2 = start_ind2[tnum]+cnum; //Index into counts[] array

    // Make all threads in a block wait till reqs[i*8+6] => Ids for same (train,class) pairs
    // Is equal to its count as seen in counts[] array
    // Ex: If reqs 1,4,5 have same (train,class), execute 1 first then 4 then 5
    if(tid == 0){
        while(1){
            atomicCAS((unsigned *)&counts[i2],counts[i2],counts[i2]);
            //atomicCAS((unsigned *)&reqs[rid_base+6],reqs[rid_base+6],reqs[rid_base+6]);
            if(reqs[rid_base+6] <= counts[i2]){
                break;
            }
        }
    } 
    __syncthreads();

    //Index into huge array
    int off = src_dst;
    int base = start_ind1[tnum]+cnum*(off);

    // Check available seats for req's stations
    if(tid >=src && tid < dst){
        if(huge_arr_vol[base+tid] < seats){  
            atomicInc(&cc,55);
        }
    }
    __syncthreads();
    //If cc increased, then booking failed
    //Else subtract seats from free seats for all stations
    if(tid >=src && tid < dst){
        if(cc == 0){  
            atomicSub((unsigned int *)&huge_arr_vol[base+tid],seats);
        }
    }
    __syncthreads();
    //Thread 0 gets responsibitly of changing counts[] value for train,class pair
    if(tid == 0){
        if(cc != 0){
            status[reqs[rid_base]] = false;
        }
        atomicInc((unsigned int *)&counts[i2],500000);
    }
    __syncthreads();
}

int main()
{
    int num_trains;
    int batches;
    scanf("%d", &num_trains); 
    int c1 = 0;     //Sum of number of classes of each train
    int c2 = 0;     //Sum of number of classes of each train*(dst-stc)
    int *temp1 = (int *)malloc(num_trains*sizeof(int));     //Prefix sum, to index array of classes
    int *temp2 = (int *)malloc(num_trains*sizeof(int));     //Prefix sum, to index array of classes*(dst-src)
    temp1[0] = 0;
    temp2[0] = 0;
    struct TrainInfo *trains = new TrainInfo[num_trains];
    for(int i = 0; i < num_trains; ++i){
        int t_num,m,src,dst;
        scanf("%d", &t_num); 
        scanf("%d", &m); 
        scanf("%d", &src); 
        scanf("%d", &dst);
        trains[t_num].t_no = t_num;
        trains[t_num].num_classes = m;
        trains[t_num].src = min(src,dst);
        trains[t_num].dst = max(src,dst);
        trains[t_num].class_cap = new int[m];

        temp1[t_num] = c1;
        temp2[t_num] = c2;
        c1+=m*(trains[t_num].dst-trains[t_num].src);
        c2+=m;
        
        for(int j = 0; j < m; ++j){
            int cnum,ccap;
            scanf("%d", &cnum); 
            scanf("%d", &ccap); 
            trains[t_num].class_cap[cnum] = ccap;
        }
    }
    int *start_indices_1;       //For gpu
    int *start_indices_2;       

    int *huge_arr = (int *)malloc(c1*sizeof(int));  //Contains an element each for capacity of each class of each train at each station
    int p = 0;
    for(int i = 0; i < num_trains; ++i){
        for(int j = 0; j < trains[i].num_classes; ++j){
            for(int k = 0; k < trains[i].dst-trains[i].src; ++k){
                huge_arr[p] = trains[i].class_cap[j];
                //printf("%d ",huge_arr[p]);
                p++;
            }
            
        }
    }//printf("\n");
    int *d_huge_arr;    //For gpu
    //assert(c1==p);
    cudaMalloc(&d_huge_arr,p*sizeof(int));
    cudaMemcpy(d_huge_arr, huge_arr, p*sizeof(int), cudaMemcpyHostToDevice);
    free(huge_arr);
    setHugeArr<<<ceil(float(p)/1024),1024>>>(d_huge_arr,p);
    cudaFree(d_huge_arr);
    cudaMalloc(&start_indices_1,num_trains*sizeof(int));
    cudaMalloc(&start_indices_2,num_trains*sizeof(int));
    cudaMemcpy(start_indices_1, temp1, num_trains*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(start_indices_2, temp2, num_trains*sizeof(int), cudaMemcpyHostToDevice);
    
    /*int bl = ceil(float(num_trains)/1024);
    cout << "a2 " << bl << endl;
    initC<<<1,1>>>();
    prefix_sum<<<bl, 1024>>>(start_indices_1,num_trains);
    cudaDeviceSynchronize();
    cout << "a4" << endl;
    initC<<<1,1>>>();
    prefix_sum<<<bl, 1024>>>(start_indices_2,num_trains);
    
    cudaDeviceSynchronize();
    cout << "a1" << endl;*/
    /*cudaMemcpy(temp1,start_indices_1, num_trains*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(temp2,start_indices_2,  num_trains*sizeof(int), cudaMemcpyDeviceToHost);
        for(int i = 0; i< num_trains; ++i){
            printf("%d,",temp1[i]);
        }
        printf("\n");
        for(int i = 0; i< num_trains; ++i){
            printf("%d,",temp2[i]);
        }
        printf("\n");*/
    
    //int *counts;
    //cudaMalloc(&counts, c2*sizeof(int));
    
    scanf("%d",&batches);
    for(int i = 0 ; i < batches; ++i){
        int num_reqs;
        int *batchInfo,*batchInfo_d;
        scanf("%d",&num_reqs);
        bool *status = (bool*)malloc(num_reqs*sizeof(bool));
        bool *d_stat;                                        //status to GPU
        batchInfo = (int *)malloc(8*num_reqs*sizeof(int));
        map<pair<int,int>, int> reqCount = map<pair<int,int>, int>();   //Keep track of count of (train,class) pairs

        for(int j = 0 ; j < num_reqs; ++j){
            scanf("%d",&batchInfo[8*j]) ;
            scanf("%d",&batchInfo[8*j+1]) ;
            scanf("%d",&batchInfo[8*j+2]) ;
            int src1,dst1;
            scanf("%d",&src1);
            scanf("%d",&dst1);
            batchInfo[8*j+3] = min(src1,dst1)-trains[batchInfo[8*j+1]].src;
            batchInfo[8*j+4] = max(src1,dst1)-trains[batchInfo[8*j+1]].src;
            batchInfo[8*j+7] = trains[batchInfo[8*j+1]].dst-trains[batchInfo[8*j+1]].src;
            scanf("%d",&batchInfo[8*j+5]) ;
            pair<int,int> p = make_pair(batchInfo[8*j+1],batchInfo[8*j+2]);
            batchInfo[8*j+6] = reqCount[p];
            reqCount[p]++;
        }
        
        cudaMalloc(&batchInfo_d,8*num_reqs*sizeof(int));
        cudaMemcpy(batchInfo_d, batchInfo, 8*num_reqs*sizeof(int), cudaMemcpyHostToDevice);
        setReqs<<<ceil(float(num_reqs)/128),1024>>>(batchInfo_d,num_reqs);
        cudaMalloc(&d_stat,num_reqs*sizeof(bool));

        setAllTrue<<<ceil(float(num_reqs)/1024),1024>>>(d_stat,num_reqs);
        setAllZeroes<<<ceil(float(c2)/1024),1024>>>(c2);
        cudaDeviceSynchronize();
        
        // Each block services 1 request
        // Each thread reduces free seats for each station in the request
        BookIt<<<num_reqs,51>>>(start_indices_1,start_indices_2,d_stat);
        cudaMemcpy(status,d_stat,num_reqs*sizeof(bool), cudaMemcpyDeviceToHost);

        cudaFree(batchInfo_d);
        cudaFree(d_stat);

        int suc=0,fal =0,tc = 0;
        for(int l = 0; l< num_reqs; ++l){
            if(status[l]){
                suc++;
                tc+=batchInfo[8*l+5]*((batchInfo[8*l+4]-batchInfo[8*l+3]));
                printf("success\n");
            }
            else{
                fal++;
                printf("failure\n");
            }
        }
        printf("%d %d\n%d\n",suc,fal,tc);
        free(batchInfo);
        free(status);
    }
}
