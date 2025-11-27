#ifndef PARAMS_H
#define PARAMS_H
#include "globals.h"
#include "my_math.h"
#include "replicated_secret_sharing.h"
#include "tensor.h"
#include "party3pc.h"

// follow the pattern of debug level 2
template <typename T>
struct pc_cmp_param
{
    T self_r0;
    T self_r1;
    T r_with_pre;
    T r_with_next;
    Tensor<T> r_with_pre_bits;
    Tensor<T> r_with_next_bits;

    T round1_r;
    T round2_r;
    Tensor<T> round1_real_table_with_pre;
    Tensor<T> round1_real_table_with_next;
    Tensor<T> round2_real_table_with_pre;
    Tensor<T> round2_real_table_with_next;
};

template <typename T>
class LUT_Param
{
public:
    Tensor<T> table;

    uint32_t table_size;
    T self_r0;
    T self_r1;
    T r_with_pre;
    T r_with_next;

    Tensor<T> onehot_table_with_pre;
    Tensor<T> onehot_table_with_next;

    void init(uint32_t table_size)
    {
        this->table_size = table_size;
        table.allocate({table_size});

        self_r0 = 0;
        self_r1 = 0;
        r_with_pre = 0;
        r_with_next = 0;

        onehot_table_with_pre.allocate({table_size});
        onehot_table_with_next.allocate({table_size});

        onehot_table_with_pre.zero();
        onehot_table_with_next.zero();

        onehot_table_with_pre.data[0] = 1;
    }
};

template <typename T>
class EQUAL_Param
{
public:
    RSSTensor<uint8_t> table_1;
    RSSTensor<uint8_t> table_2;
    uint32_t t1_size;
    uint32_t t2_size;

    T r_11;
    T r_12;
    uint8_t r_21;
    uint8_t r_22;

    void init(uint32_t t1_size, uint32_t t2_size)
    {
        this->t1_size = t1_size;
        this->t2_size = t2_size;
        table_1.allocate({t1_size});
        table_2.allocate({t2_size});

        r_11 = 0;
        r_12 = 0;
        r_21 = 0;
        r_22 = 0;
    }
};

// template <typename T>
// class GREATER_EQUAL_Param
// {
// public:
//     ASSTensor<uint8_t> table_1;
//     ASSTensor<uint8_t> table_2;
//     uint32_t t1_size;
//     uint32_t t2_size;

//     T r_1;
//     uint8_t r_2;
//     T r;
//     uint8_t msb_2_l_r;

//     void init(uint32_t t1_size, uint32_t t2_size)
//     {
//         this->t1_size = t1_size;
//         this->t2_size = t2_size;
//         table_1.allocate({t1_size});
//         table_2.allocate({t2_size});

//         r_1 = 0;
//         r_2 = 0;
//         r = 0;
//         msb_2_l_r = 0;
//     }
// };


template <typename T>
class Parameters
{
public:
    struct pc_cmp_param<T> pc_cmp;
    LUT_Param<T> nexp2_param;
    LUT_Param<T> nexpb_param;
    LUT_Param<T> inv_param;
    LUT_Param<T> sqrt_nexpb_param;
    LUT_Param<T> rsqrt_param;
    LUT_Param<T> gelu_param;
    EQUAL_Param<T> equal_param;
    // GREATER_EQUAL_Param<T> greater_param;
    int party_id;

    static constexpr auto scale_bit = []
    {
        if constexpr (std::is_same_v<T, uint64_t>)
        {
            return FLOAT_PRECISION_64;
        }
        else if constexpr (std::is_same_v<T, uint32_t>)
        {
            return FLOAT_PRECISION_32;
        }
        else if constexpr (std::is_same_v<T, uint16_t>)
        {
            return FLOAT_PRECISION_16;
        }
        else
        {
            return 1;
        }
    }();

    Parameters()
    {
        this->party_id = Party3PC::getInstance().party_id;
    }

    void init_pc_cmp()
    {
        pc_cmp.self_r0 = 0;
        pc_cmp.self_r1 = 0;

        pc_cmp.r_with_pre = 0;
        pc_cmp.r_with_next = 0;

        pc_cmp.r_with_pre_bits.allocate({8 * sizeof(T)});
        pc_cmp.r_with_next_bits.allocate({8 * sizeof(T)});

        pc_cmp.r_with_pre_bits.zero();
        pc_cmp.r_with_next_bits.fill(1);
        pc_cmp.r_with_next_bits.data[0] = 0;

        pc_cmp.round1_r = 0;
        pc_cmp.round2_r = 0;

        uint32_t double_bit_length = 2 * 8 * sizeof(T);
        pc_cmp.round1_real_table_with_pre.allocate({double_bit_length});
        pc_cmp.round1_real_table_with_next.allocate({double_bit_length});
        pc_cmp.round2_real_table_with_pre.allocate({double_bit_length});
        pc_cmp.round2_real_table_with_next.allocate({double_bit_length});

        pc_cmp.round1_real_table_with_pre.zero();
        pc_cmp.round1_real_table_with_next.zero();
        pc_cmp.round2_real_table_with_pre.zero();
        pc_cmp.round2_real_table_with_next.zero();

        pc_cmp.round1_real_table_with_pre.data[0] = 1;
        pc_cmp.round2_real_table_with_pre.data[0] = 1;
    }

    void init_nexp2()
    {
        uint32_t table_size = 2 * scale_bit + 1;
        nexp2_param.init(table_size);
        for (int i = 0; i < table_size; i++)
        {
            nexp2_param.table.data[i] = 1 << (scale_bit - i + scale_bit);
        }
    }

    void init_nexpb()
    {
        uint32_t table_size = (int)(log(pow(2, 2 * scale_bit)) / log(SCALE_BASE)) + 1;
        nexpb_param.init(table_size);
        for(int i = 0; i < table_size; i++)
        {
            nexpb_param.table.data[i] = (T)((pow(2, scale_bit) / pow(SCALE_BASE, i)) * (1 << scale_bit));
        }
    }

    void init_inv()
    {
        uint32_t table_size = (uint32_t)ceil((1.0  - 1.0 / SCALE_BASE) * (1 << scale_bit));  // 1 / b * (2 ^ f)
        float table_scale = 1 << scale_bit;
        inv_param.init(table_size);

        for(int i = 0; i < table_size; i++)
        {
            float real_value = i / table_scale + 1.0 / SCALE_BASE;
            inv_param.table.data[i] = (T)((1 / real_value) * (1 << scale_bit));
        }
    }

    void init_rsqrt()
    {
        uint32_t table_size = (uint32_t)ceil((1.0  - 1.0 / SCALE_BASE) * (1 << scale_bit));  // 0.75 * (2 ^ f)
        float table_scale = 1 << scale_bit;
        rsqrt_param.init(table_size);

        for(int i = 0; i < table_size; i++)
        {
            float real_value = i / table_scale + 1.0 / SCALE_BASE;
            rsqrt_param.table.data[i] = (T)(sqrt(1 / real_value) * (1 << scale_bit));
        }

        uint32_t sqrt_nexpb_table_size = (int)(log(pow(2, 2 * scale_bit)) / log(SCALE_BASE)) + 1;;
        sqrt_nexpb_param.init(sqrt_nexpb_table_size);
        for(int i = 0; i < sqrt_nexpb_table_size; i++)
        {
            sqrt_nexpb_param.table.data[i] = (T)(sqrt(pow(2, scale_bit) / pow(SCALE_BASE, i)) * (1 << scale_bit));
        }
    }

    void init_gelu()
    {
        // table_scale_bit = 6;
        uint32_t table_size = 4 * (1 << GELU_TABLE_PRECISION);
        float table_scale = 1 << GELU_TABLE_PRECISION;
        gelu_param.init(table_size);

        for (int i = 0; i < table_size; i++)
        {
            float real_value = i / table_scale;
            gelu_param.table.data[i] = (T)((ReLU(real_value) - GeLU(real_value)) * (1 << scale_bit));
        }
    }

    void init_equal()
    {
        if(sizeof(T) < 3){
            equal_param.init((uint32_t)(32 * sizeof(T)), (uint32_t)1);
        }
        else{
            equal_param.init((uint32_t)(32 * sizeof(T)), (uint32_t)(1 << (sizeof(T) - 3)));
        }
        
        equal_param.table_1.zeros();
        equal_param.table_2.zeros();

        if(party_id == 0)
        {
            for (int i = 0; i < sizeof(T); i++)
            {
                equal_param.table_1.first.data[32*i] = 128;
            }
            equal_param.table_2.first.data[0] = 128;
        }
        else if(party_id == 2)
        {
            for (int i = 0; i < sizeof(T); i++)
            {
                equal_param.table_1.second.data[32*i] = 128;
            }
            equal_param.table_2.second.data[0] = 128;
        }
    }

    // void init_greater_equal()
    // {
    //     if(sizeof(T) < 3){
    //         greater_param.init((uint32_t)(32 * sizeof(T)), (uint32_t)1);
    //     }
    //     else{
    //         greater_param.init((uint32_t)(32 * sizeof(T)), (uint32_t)(1 << (sizeof(T) - 3)));
    //     }
        
    //     greater_param.r_1 = 3 * (1 - party_id); // r = 3;
    //     greater_param.r = greater_param.r_1 + (1 - party_id);

    //     if(party_id == 0)
    //     {
    //         greater_param.msb_2_l_r = (- greater_param.r) >> 8 * sizeof(T) - 1 & 1;
    //     }
    //     else
    //     {
    //         greater_param.msb_2_l_r = 0;
    //     }

    //     greater_param.table_1.zeros();
    //     greater_param.table_2.zeros();

    //     for (int i = 0; i < sizeof(T); i++)
    //     {
    //         if(i == 0)
    //         {
    //             greater_param.table_1.value.data[32*i] = 16 * (1 - party_id); 
    //         }
    //         else
    //         {
    //             greater_param.table_1.value.data[32*i] = 128 * (1 - party_id);
    //         }
    //     }

    //     greater_param.table_2.value.data[0] = 128 * (1 - party_id);
    // }

    void init_all()
    {
        init_pc_cmp();
        init_nexp2();
        init_nexpb();
        init_inv();
        init_rsqrt();
        init_gelu();
        init_equal();
    }
};

#endif // PARAMS_H