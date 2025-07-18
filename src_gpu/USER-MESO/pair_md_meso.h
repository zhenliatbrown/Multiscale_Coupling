#ifdef PAIR_CLASS

PairStyle(md/meso,MesoPairMD)

#else

#ifndef LMP_PAIR_MD_MESO_H
#define LMP_PAIR_MD_MESO_H

#include "pair.h"
#include "meso.h"

namespace LAMMPS_NS
{
    namespace MD_COEFFICIENTS
    {
        const static int p_epsilon = 0;
        const static int p_sigma = 1;
        const static int p_cut = 2;
        const static int p_cutsq = 3;
        const static int p_lj1 = 4;
        const static int p_lj2 = 5;
        const static int p_lj3 = 6;
        const static int p_lj4 = 7;
        const static int p_offset = 8;
        const static int n_coeff = 9;
    }

    class MesoPairMD : public Pair, protected MesoPointers
    {
    public:
        MesoPairMD(class LAMMPS*);
        virtual ~MesoPairMD();

        void compute(int, int) override;
        void compute_bulk(int, int) override;
        void compute_border(int, int) override;
        
        void settings(int, char **) override;
        void coeff(int, char **) override;
        void init_style() override;
        double init_one(int, int) override;

        void write_restart(FILE *) override;
        void read_restart(FILE *) override;
        void write_restart_settings(FILE *) override;
        void read_restart_settings(FILE *) override;

        double single(int, int, int, int, double, double, double, double& ) override;
        

    protected:
        int seed;
        bool coeff_ready;
        DeviceScalar<r64> dev_coefficients;
        std::vector<r64> coeff_table;

        double cut_global;
        double **cut;
        // double **cutsq;
        double **epsilon;
        double **sigma;
        double **lj1, **lj2, **lj3, **lj4, **offset;
        // double **cut_respa;

        virtual void allocate();
        virtual void prepare_coeff();
        virtual void compute_kernel(int, int, int, int);
        virtual uint seed_now();
    };
}

#endif
#endif

