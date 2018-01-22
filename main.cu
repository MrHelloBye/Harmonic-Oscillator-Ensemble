//main.cpp
#include "vose.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cstdio>

using namespace std;

# define PI 3.14159265358979323846

__device__ double potential(double*);
__device__ double potential_osc(double*);
__device__ double potential3(double*);
__device__ double kinetic(double*);
__device__ double Hamiltonian(double*, double*, unsigned int, double);
__global__ void update_state(double*, double*, unsigned int, double, double);
void write_state(FILE*, double, double*, double*, unsigned int);

const double length = 10;
double mass = 0.005;

int main(int argc, char *argv[])
{
    auto seed = chrono::high_resolution_clock::now().time_since_epoch().count();
    mt19937 mt(seed); // Seeds Mersenne Twister with Device RNG
    
    bool random_sampling = false;
    
    // Define the density array for position
    unsigned int grid_size = 10000; //number of grid points
    unsigned int particle_number = 64;
    
    double *positions;
    cudaMallocManaged(&positions, particle_number*sizeof(double));
    
    // Random sampling
    if (random_sampling)
    {
        double *density = new double[grid_size];
        double arg_scale = 2*PI/grid_size;
        double norm = 0;
        for (unsigned int i = 0; i<grid_size; i++)
        {
            density[i] = 0.5*sin(arg_scale*i)+1;
            norm += density[i];
        }
    
        //Normalize position density
        for (unsigned int i = 0; i<grid_size; i++)
        {
            density[i] /= norm;
        }
        
        // Sample locations
        vose *pos_sampler = new vose(density, grid_size, mt);
    
        for (unsigned int i=0; i<particle_number; i++)
        {   
            positions[i] = pos_sampler->alias_method()*length/grid_size;
        }

        //Sort locations (just so IC is nicer)
        sort(positions, positions+particle_number);
    }
    
    // Even spacing
    else
    {
        double spacing = length/(particle_number+1);
        for (unsigned int i=0; i<particle_number; i++)
        {
            positions[i] = spacing*(i+1);
        }
    }
    
    
    // Define momentum density
    double *mom_density = new double[grid_size];
    double Temperature = 100;
    double arg_scale = -1/(2*mass*Temperature);
    
    double norm = 0;
    for (unsigned int i = -floor(grid_size/2); i<floor(grid_size/2); i++)
    {
        mom_density[i] = exp(arg_scale*i*i);
        norm += mom_density[i];
    }
    
    //Normalize momentum density
    for (unsigned int i = 0; i<grid_size; i++)
    {
        mom_density[i] /= norm;
    }
    
    vose *mom_sampler = new vose(mom_density, grid_size, mt);
    
    double *momenta;
    cudaMallocManaged(&momenta, particle_number*sizeof(double));
    
    double mom_scale = 0.0005;
    unsigned int j;
    for (unsigned int i=0; i<particle_number; i++)
    {   
        //j = mom_sampler->alias_method(); //momentum index
        //momenta[i] = (j-float(grid_size)/2)*mom_scale; //convert j to momentum
        momenta[i] = 0;
    }
    
    /*--------------------------------------------*/
    
	//Create output file
	FILE *output_file;
	output_file = fopen("output.tsv", "w");
	fprintf(output_file, "Time\tPositions\tMomenta\n");
    
    // Hamiltonian solution part
    double dt = 0.01;
    double max_time = 1200;
    
    int counter = 0;
    for (double t = 0; t<max_time; t+=dt)
    {
        cout << t << endl;
        update_state<<<1,256>>>(positions, momenta, particle_number, dt, t);
        counter++;
        if (false)//(counter % 10 == 0)
        {
            // Wait for GPU to finish before accessing on host
            // Otherwise you get a Bus error: 10
            cudaDeviceSynchronize();
            
            write_state(output_file, t, positions, momenta, particle_number);
        }
    }
    
    cudaFree(positions);
    cudaFree(momenta);
    
    return 0;
}

__device__
double potential(double *positions, unsigned int particle_number)
{
    double k = 0.1;
    double potential_energy = 0;
    double scale = k/2;
    double spacing = length/particle_number;
    
    for (unsigned int i = 0; i<=particle_number; i++)
    {
        if (i==0)
            potential_energy += scale*pow(positions[0]-spacing,2);
        else if (i==particle_number)
            potential_energy += scale*pow(length-positions[i-1]-spacing,2);
        else
            potential_energy += scale*pow(positions[i]-positions[i-1]-spacing,2);
    }
    
    return potential_energy;
}

__device__
double potential_osc(double *positions, unsigned int particle_number, double t)
{
    double k = 0.1;
    double potential_energy = 0;
    double scale = k/2;
    double spacing = length/particle_number;
    
    for (unsigned int i = 0; i<=particle_number; i++)
    {
        if (i==0)
        {
            double driver = 10*sin(2*PI*t/10);
            potential_energy += scale*pow(positions[0]-spacing-driver,2);
        }
        else if (i==particle_number)
            potential_energy += scale*pow(length-positions[i-1]-spacing,2);
        else
            potential_energy += scale*pow(positions[i]-positions[i-1]-spacing,2);
    }
    
    return potential_energy;
}

__device__
double potential3(double *positions, unsigned int particle_number)
{
    double k = 0.1;
    double potential_energy = 0;
    double scale = k/2;
    double spacing = length/particle_number;
    
    for (unsigned int i = 0; i<=particle_number; i++)
    {
        if (i==0)
            potential_energy += scale*fabs(pow(positions[0]-spacing,3));
        else if (i==particle_number)
            potential_energy += scale*fabs(pow(length-positions[i-1]-spacing,3));
        else
            potential_energy += scale*fabs(pow(positions[i]-positions[i-1]-spacing,3));
    }
    
    return potential_energy;
}

__device__
double kinetic(double *momenta, unsigned int particle_number)
{
    double kinetic_energy = 0;
    double mass = 10;
    double scale = 1/(2*mass);
    
    for (unsigned int i = 0; i<particle_number; i++)
    {
        kinetic_energy += scale*pow(momenta[i],2);
    }
    
    return kinetic_energy;
}

__device__
double Hamiltonian(double *positions,double *momenta,unsigned int particle_number, double t)
{
    return potential_osc(positions,particle_number, t) + kinetic(momenta,particle_number);
}

//Gauss-Seidel I think
__global__
void update_state(double *pos, double *mom, unsigned int part_num, double dt, double t)
{
    double dx = 0.001, dp = 0.001; //finite differences
    double H_pplus, H_pminus, H_xplus, H_xminus;
    double p_grad, x_grad;
    
    int index = threadIdx.x;
    int stride = blockDim.x;
    
    for (int i = index; i<part_num; i += stride)
    {
        //Calculate phase space gradient
        pos[i] += dx;
        H_xplus = Hamiltonian(pos, mom, part_num, t);
        pos[i] -= 2*dx;
        H_xminus = Hamiltonian(pos, mom, part_num, t);
        pos[i] += dx;
        x_grad = (H_xplus-H_xminus)/(2*dx);
        
        //Update momentum
        mom[i] -= dt*x_grad;
        
        mom[i] += dp;
        H_pplus = Hamiltonian(pos, mom, part_num, t);
        mom[i] -= 2*dp;
        H_pminus = Hamiltonian(pos, mom, part_num, t);
        mom[i] += dp;
        p_grad = (H_pplus-H_pminus)/(2*dp);
        
        //Update position
        pos[i] += dt*p_grad;
        
    }
    //cout << Hamiltonian(pos, mom, part_num) << endl;
}

void write_state(FILE *fp, double t, double *pos, double *mom, unsigned int part_num)
{
    cout << "check 1\n";
    
    //Write time
    fprintf(fp,"%f\t",t);
    
    cout << "check 2\n";
    
    //Write positions
    for (unsigned int i = 0; i<part_num; i++)
    {
        fprintf(fp,"%f\t",pos[i]);
    }
    
    cout << "check 3\n";
    
    //Write momenta
    for (unsigned int i = 0; i<part_num; i++)
    {
        fprintf(fp,"%f",mom[i]);
        if (i<part_num-1)
            fprintf(fp,"\t");
    }
    
    cout << "check 4\n";
    
    fprintf(fp,"\n");
    
}