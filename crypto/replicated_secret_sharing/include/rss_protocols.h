#pragma once

#include <fstream>
#include <cmath>
#include "globals.h"
#include "params.h"
#include "rss_protocols.h"
#include "ass_protocols.h"
#include "party3pc.h"

namespace rss_protocols
{
    namespace debug
    {
        template <typename T>
        void openPrintReal(RSSTensor<T> &x);

        template <typename T>
        void openPrintReal(RSSTensor<T> &x, int index);

        template <typename T>
        void openPrintReal(RSSTensor<T> &x, std::vector<int> index);

        template <typename T>
        void openPrintReal(RSSTensor<T> &x, int start, int end);

        template <typename T>
        void printRealToFile(RSSTensor<T> &x, const std::string &file_name);
    }

    namespace utils
    {
        template <typename T>
        void RSSMatMul(RSSTensor<T> &x, RSSTensor<T> &y, Tensor<T> &res, const uint32_t broadcast_size,
                       const uint32_t common_size, const uint32_t row, const uint32_t common_dim, const uint32_t col,
                       const bool is_b_broadcast);

        template <typename T>
        void RSSMatMul(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &res, const uint32_t broadcast_size,
                       const uint32_t common_size, const uint32_t row, const uint32_t common_dim, const uint32_t col,
                       const bool is_b_broadcast);

        template <typename T>
        void RSSMatMul(Tensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, const uint32_t broadcast_size,
                       const uint32_t common_size, const uint32_t row, const uint32_t common_dim, const uint32_t col,
                       const bool is_b_broadcast);

        template <typename T>
        void privateCompare(Tensor<T> &x_with_pre, Tensor<T> &x_with_next, Tensor<T> &res_with_pre,
                            Tensor<T> &res_with_next, Parameters<T> &parameter);

        template <typename T>
        void getk(RSSTensor<T> &x, RSSTensor<T> &k, Parameters<T> &parameter, bool malicious = false);

        template <typename T>
        void gelu_same_scale(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious = false);
    }

    template <typename T>
    void restore(RSSTensor<T> &x, Tensor<T> &res, bool malicious = false);

    template <typename T>
    void restore_bit(RSSTensor<T> &x, Tensor<T> &res, bool malicious = false);

    template <typename T>
    void reconstruct_to(int target_id, RSSTensor<T> &x, Tensor<T> &res = NULL, bool malicious = false);

    template <typename T>
    void share(Tensor<T> &x, RSSTensor<T> &res);

    template <typename T>
    void recv_shares_from(int source_id, RSSTensor<T> &res);

    template <typename T>
    void coin(std::vector<uint32_t> shape, RSSTensor<T> &res);

    template <typename T>
    void reshare(Tensor<T> &x, RSSTensor<T> &res);

    template <typename T>
    RSSTensor<T> reshare(Tensor<T> &x);

    template <typename T>
    void add(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res);

    template <typename T>
    void add(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &res);

    template <typename T>
    void add(RSSTensor<T> &x, T y, RSSTensor<T> &res);

    template <typename T>
    void sub(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res);

    template <typename T>
    void sub(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &res);

    template <typename T>
    void sub(RSSTensor<T> &x, T y, RSSTensor<T> &res);

    template <typename T>
    void sub(T x, RSSTensor<T> &y, RSSTensor<T> &res);

    template <typename T>
    void mulConst(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &res);

    template <typename T>
    void mulConst(RSSTensor<T> &x, T y, RSSTensor<T> &res);

    template <typename T>
    void mulConstAddBias(RSSTensor<T> &x, T y, T bias, RSSTensor<T> &res);

    template <typename T>
    void mulConstAddBias(RSSTensor<T> &x, T y, RSSTensor<T> &bias, RSSTensor<T> &res);

    template <typename T>
    void mulConstAddBias(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &bias, RSSTensor<T> &res);

    template <typename T>
    void mulConstSubBias(RSSTensor<T> &x, T y, T bias, RSSTensor<T> &res);

    template <typename T>
    void mulConstSubBias(RSSTensor<T> &x, T y, RSSTensor<T> &bias, RSSTensor<T> &res);

    template <typename T>
    void mulConstSubBias(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &bias, RSSTensor<T> &res);

    template <typename T>
    void mul(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, bool needTrunc = false, bool malicious = false);

    template <typename T>
    void mul(RSSTensor<T> &x, std::pair<T, T> &y, RSSTensor<T> &res);

    template <typename T>
    void square(RSSTensor<T> &x, RSSTensor<T> &res, bool needTrunc = false, bool malicious = false);

    template <typename T>
    void matMul(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, bool needTrunc = false, bool malicious = false);

    template <typename T>
    void truncate(RSSTensor<T> &x, RSSTensor<T> &res, size_t scale, bool malicious = false);

    template <typename T>
    void truncate(RSSTensor<T> &x, size_t scale, bool malicious = false);

    template <typename T>
    void truncate(RSSTensor<T> &x, bool malicious = false);

    template <typename T>
    void checkZero(RSSTensor<T> &x);

    template <typename T>
    void macCheck(const RSSTensor<T> &x, const RSSTensor<T> &mx, const std::pair<T, T> &mac_key);

    template <typename T>
    void macCheck(RSSTensor<T> &x, RSSTensor<T> &mx, RSSTensor<T> &mac_key);

    template <typename T>
    void macCheckSimulate(uint32_t size);

    template <typename T>
    void pc_msb(RSSTensor<T> &x, Tensor<T> &res_with_pre, Tensor<T> &res_with_next,
                Parameters<T> &parameter, const uint32_t size, bool malicious = false);

    template <typename T>
    void nonNegative(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameter, bool isFloat = true, bool malicious = false);

    template <typename T>
    void greaterEqual(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, Parameters<T> &parameter, bool isFloat = true, bool malicious = false);

    template <typename T>
    void b2a(RSSTensor<T> &x, RSSTensor<T> &res, bool malicious = false);

    template <typename T>
    void equal_judge(Tensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameter, bool malicious = false);

    template <typename T>
    void table_Equal(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, Parameters<T> &parameter, 
                     bool is_b2a, bool isFloat = true, bool malicious = false);

    template <typename T>
    void greater_judge(Tensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameter, bool malicious = false);

    template <typename T>
    void table_greaterEqual(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, Parameters<T> &parameter, 
                            bool is_b2a, bool isFloat = true, bool malicious = false);

    template <typename T>
    void select(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, Parameters<T> &parameter, bool malicious = false);

    template <typename T>
    void select(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, uint32_t y_num, Parameters<T> &parameter, bool malicious = false);

    template <typename T>
    void lut(RSSTensor<T> &x, RSSTensor<T> &res, LUT_Param<T> &parameter, bool malicious = false);

    template <typename T>
    void lut(RSSTensor<T> &x, RSSTensor<T> &res1, RSSTensor<T> &res2, LUT_Param<T> &parameter1, LUT_Param<T> &parameter2, bool malicious = false);

    template <typename T>
    void inv(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious = false);

    template <typename T>
    void rsqrt(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious = false);

    template <typename T>
    void gelu(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious = false);

    template <typename T>
    void max_last_dim(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious = false);

    template <typename T>
    void neg_exp(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &Parameters, bool malicious = false);

    template <typename T>
    void softmax_forward(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameter, bool malicious = false);

    template <typename T, typename U>
    void downcast(RSSTensor<T> &x, RSSTensor<U> &res);

    template <typename U, typename T>
    void upcast(RSSTensor<U> &x, RSSTensor<T> &res, int party_id, bool malicious = false);
}

template <typename T>
void rss_protocols::debug::openPrintReal(RSSTensor<T> &x)
{
    openPrintReal(x, 0, x.size());
}

template <typename T>
void rss_protocols::debug::openPrintReal(RSSTensor<T> &x, int index)
{
    openPrintReal(x, index, index + 1);
}

template <typename T>
void rss_protocols::debug::openPrintReal(RSSTensor<T> &x, std::vector<int> index)
{
    Tensor<T> real_res(x.shape);
    restore(x, real_res);

    if (Party3PC::getInstance().party_id == 0)
    {
        for (int i : index)
        {
            always_assert(i < x.size());
            std::cout << (float)(long)real_res.data[i] / (1 << x.float_scale_bit) << ", ";
        }
        std::cout << std::endl;
    }
}

template <typename T>
void rss_protocols::debug::openPrintReal(RSSTensor<T> &x, int start, int end)
{
    always_assert(start < end);
    Tensor<T> real_res(x.shape);
    restore(x, real_res);

    if (Party3PC::getInstance().party_id == 0)
    {
        for (int i = start; i < end; i++)
        {
            if constexpr (std::is_same_v<T, uint32_t>)
            {
                std::cout << (float)(int)real_res.data[i] / (1 << x.float_scale_bit) << ", ";
            }
            else
            {
                std::cout << (double)(long)real_res.data[i] / (1 << x.float_scale_bit) << ", ";
            }
        }
        std::cout << std::endl;
    }
}

template <typename T>
void rss_protocols::debug::printRealToFile(RSSTensor<T> &x, const std::string &file_name)
{
    Tensor<T> real_res(x.shape);
    restore(x, real_res);

    if (Party3PC::getInstance().party_id == 0)
    {
        std::ofstream outFile;
        outFile.open(file_name);
        outFile << "[";
        for (int i = 0; i < x.size(); i++)
        {
            if constexpr (std::is_same_v<T, uint32_t>)
            {
                outFile << ((float)(int)real_res.data[i] / (1 << x.float_scale_bit)) << ", ";
            }
            else
            {
                outFile << ((double)(long)real_res.data[i] / (1 << x.float_scale_bit)) << ", ";
            }
        }
        outFile << "]" << std::endl;
    }
}

template <typename T>
void rss_protocols::utils::RSSMatMul(RSSTensor<T> &x, RSSTensor<T> &y, Tensor<T> &res, const uint32_t broadcast_size,
                                     const uint32_t common_size, const uint32_t row, const uint32_t common_dim, const uint32_t col,
                                     const bool is_b_broadcast)
{
    uint32_t index, index_a, index_b;
    if (is_b_broadcast)
    {
        for (uint32_t i = 0; i < broadcast_size; i++)
        {
            for (uint32_t j = 0; j < common_size; j++)
            {
#pragma omp parallel for
                for (uint32_t k = 0; k < row; k++)
                {
                    for (uint32_t l = 0; l < col; l++)
                    {
                        index = i * common_size * row * col + j * row * col + k * col + l;
                        index_a = i * common_size * row * common_dim + j * row * common_dim + k * common_dim;
                        index_b = j * common_dim * col + l;

                        res.data[index] = 0;
                        for (uint32_t m = 0; m < common_dim; m++)
                        {
                            res.data[index] += x.first.data[index_a + m] * y.first.data[index_b + m * col] + x.second.data[index_a + m] * y.first.data[index_b + m * col] + x.first.data[index_a + m] * y.second.data[index_b + m * col];
                        }
                    }
                }
            }
        }
    }
    else
    {
        for (uint32_t i = 0; i < broadcast_size; i++)
        {
            for (uint32_t j = 0; j < common_size; j++)
            {
#pragma omp parallel for
                for (uint32_t k = 0; k < row; k++)
                {
                    for (uint32_t l = 0; l < col; l++)
                    {
                        index = i * common_size * row * col + j * row * col + k * col + l;
                        index_a = j * row * common_dim + k * common_dim;
                        index_b = i * common_size * common_dim * col + j * common_dim * col + l;

                        res.data[index] = 0;
                        for (uint32_t m = 0; m < common_dim; m++)
                        {
                            res.data[index] += x.first.data[index_a + m] * y.first.data[index_b + m * col] + x.second.data[index_a + m] * y.first.data[index_b + m * col] + x.first.data[index_a + m] * y.second.data[index_b + m * col];
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void rss_protocols::utils::RSSMatMul(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &res, const uint32_t broadcast_size,
                                     const uint32_t common_size, const uint32_t row, const uint32_t common_dim, const uint32_t col,
                                     const bool is_b_broadcast)
{
    uint32_t index, index_a, index_b;
    if (is_b_broadcast)
    {
        for (uint32_t i = 0; i < broadcast_size; i++)
        {
            for (uint32_t j = 0; j < common_size; j++)
            {
#pragma omp parallel for
                for (uint32_t k = 0; k < row; k++)
                {
                    for (uint32_t l = 0; l < col; l++)
                    {
                        index = i * common_size * row * col + j * row * col + k * col + l;
                        index_a = i * common_size * row * common_dim + j * row * common_dim + k * common_dim;
                        index_b = j * common_dim * col + l;

                        res.first.data[index] = 0;
                        res.second.data[index] = 0;
                        for (uint32_t m = 0; m < common_dim; m++)
                        {
                            res.first.data[index] += x.first.data[index_a + m] * y.data[index_b + m * col];
                            res.second.data[index] += x.second.data[index_a + m] * y.data[index_b + m * col];
                        }
                    }
                }
            }
        }
    }
    else
    {
        for (uint32_t i = 0; i < broadcast_size; i++)
        {
            for (uint32_t j = 0; j < common_size; j++)
            {
#pragma omp parallel for
                for (uint32_t k = 0; k < row; k++)
                {
                    for (uint32_t l = 0; l < col; l++)
                    {
                        index = i * common_size * row * col + j * row * col + k * col + l;
                        index_a = j * row * common_dim + k * common_dim;
                        index_b = i * common_size * common_dim * col + j * common_dim * col + l;

                        res.first.data[index] = 0;
                        res.second.data[index] = 0;
                        for (uint32_t m = 0; m < common_dim; m++)
                        {
                            res.first.data[index] += x.first.data[index_a + m] * y.data[index_b + m * col];
                            res.second.data[index] += x.second.data[index_a + m] * y.data[index_b + m * col];
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void rss_protocols::utils::RSSMatMul(Tensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, const uint32_t broadcast_size,
                                     const uint32_t common_size, const uint32_t row, const uint32_t common_dim, const uint32_t col,
                                     const bool is_b_broadcast)
{
    uint32_t index, index_a, index_b;
    if (is_b_broadcast)
    {
        for (uint32_t i = 0; i < broadcast_size; i++)
        {
            for (uint32_t j = 0; j < common_size; j++)
            {
#pragma omp parallel for
                for (uint32_t k = 0; k < row; k++)
                {
                    for (uint32_t l = 0; l < col; l++)
                    {
                        index = i * common_size * row * col + j * row * col + k * col + l;
                        index_a = i * common_size * row * common_dim + j * row * common_dim + k * common_dim;
                        index_b = j * common_dim * col + l;

                        res.first.data[index] = 0;
                        res.second.data[index] = 0;
                        for (uint32_t m = 0; m < common_dim; m++)
                        {
                            res.first.data[index] += x.data[index_a + m] * y.first.data[index_b + m * col];
                            res.second.data[index] += x.data[index_a + m] * y.second.data[index_b + m * col];
                        }
                    }
                }
            }
        }
    }
    else
    {
        for (uint32_t i = 0; i < broadcast_size; i++)
        {
            for (uint32_t j = 0; j < common_size; j++)
            {
#pragma omp parallel for
                for (uint32_t k = 0; k < row; k++)
                {
                    for (uint32_t l = 0; l < col; l++)
                    {
                        index = i * common_size * row * col + j * row * col + k * col + l;
                        index_a = j * row * common_dim + k * common_dim;
                        index_b = i * common_size * common_dim * col + j * common_dim * col + l;

                        res.first.data[index] = 0;
                        res.second.data[index] = 0;
                        for (uint32_t m = 0; m < common_dim; m++)
                        {
                            res.first.data[index] += x.data[index_a + m] * y.first.data[index_b + m * col];
                            res.second.data[index] += x.data[index_a + m] * y.second.data[index_b + m * col];
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void rss_protocols::restore(RSSTensor<T> &x, Tensor<T> &res, bool malicious)
{
    Party3PC &party = Party3PC::getInstance();
    x.assert_same_shape(res);
    party.send_tensor_to(party.next_party_id, x.first);

    party.recv_tensor_from(party.pre_party_id, res);

    if (malicious)
    {
        party.send_tensor_to(party.pre_party_id, x.second);
        Tensor<T> tmp(x.shape);
        party.recv_tensor_from(party.next_party_id, tmp);

        Tensor<T>::minus(tmp, res, tmp);
        tmp.assert_zero();
    }

#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        res.data[i] = res.data[i] + x.first.data[i] + x.second.data[i];
    }
}

template <typename T>
void rss_protocols::restore_bit(RSSTensor<T> &x, Tensor<T> &res, bool malicious)
{
    Party3PC &party = Party3PC::getInstance();
    x.assert_same_shape(res);
    party.send_tensor_to(party.next_party_id, x.first);

    party.recv_tensor_from(party.pre_party_id, res);

    // if (malicious)
    // {
    //     party.send_tensor_to(party.pre_party_id, x.second);
    //     Tensor<T> tmp(x.shape);
    //     party.recv_tensor_from(party.next_party_id, tmp);

    //     Tensor<T>::minus(tmp, res, tmp);
    //     tmp.assert_zero();
    // }

#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        res.data[i] = res.data[i] ^ x.first.data[i] ^ x.second.data[i];
    }   
}

template <typename T>
void rss_protocols::reconstruct_to(int target_id, RSSTensor<T> &x, Tensor<T> &res, bool malicious)
{
    Party3PC &party = Party3PC::getInstance();
    if (target_id == party.pre_party_id)
    {
        if (malicious)
        {
            party.send_tensor_to(target_id, x.second);
        }
    }
    else if (target_id == party.next_party_id)
    {
        party.send_tensor_to(target_id, x.first);
    }
    else
    {
        Tensor<T> tmp(x.shape);
        party.recv_tensor_from(party.pre_party_id, res);

        if (malicious)
        {
            party.recv_tensor_from(party.next_party_id, tmp);

            Tensor<T>::minus(tmp, tmp, res);
            tmp.assert_zero();
        }

#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.data[i] = res.data[i] + x.first.data[i] + x.second.data[i];
        }
    }
}

template <typename T>
void rss_protocols::share(Tensor<T> &x, RSSTensor<T> &res)
{
    Party3PC &party = Party3PC::getInstance();
    res.assert_same_shape(x);
    res.rand();
    Tensor<T> tmp(x.shape);
#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        tmp.data[i] = x.data[i] - res.first.data[i] - res.second.data[i];
    }

    party.send_tensor_to(party.next_party_id, res.second);
    party.send_tensor_to(party.next_party_id, tmp);

    party.send_tensor_to(party.pre_party_id, tmp);
    party.send_tensor_to(party.pre_party_id, res.first);
}

template <typename T>
void rss_protocols::recv_shares_from(int source_id, RSSTensor<T> &res)
{
    Party3PC &party = Party3PC::getInstance();
    party.recv_tensor_from(source_id, res.first);
    party.recv_tensor_from(source_id, res.second);
}

template <typename T>
void rss_protocols::coin(std::vector<uint32_t> shape, RSSTensor<T> &res)
{
}

template <typename T>
void rss_protocols::reshare(Tensor<T> &x, RSSTensor<T> &res)
{
    Party3PC &party = Party3PC::getInstance();
    res.assert_same_shape(x);
    party.send_tensor_to(party.pre_party_id, x);
    party.recv_tensor_from(party.next_party_id, res.second);
    res.first.copy(x);
}

template <typename T>
RSSTensor<T> rss_protocols::reshare(Tensor<T> &x)
{
    Party3PC &party = Party3PC::getInstance();
    Tensor<T> tmp(x.shape);
    party.send_tensor_to(party.pre_party_id, x);
    party.recv_tensor_from(party.next_party_id, tmp);
    return RSSTensor<T>(x, tmp);
}

template <typename T>
void rss_protocols::add(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res)
{
    res.assert_same_shape(x);
    res.assert_same_shape(y);

#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        res.first.data[i] = x.first.data[i] + y.first.data[i];
        res.second.data[i] = x.second.data[i] + y.second.data[i];
    }
}

template <typename T>
void rss_protocols::add(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &res)
{
    int party_id = Party3PC::getInstance().party_id;
    res.assert_same_shape(x);
    res.assert_same_shape(y);

    if (party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i] + y.data[i];
            res.second.data[i] = x.second.data[i];
        }
    }
    else if (party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i];
            res.second.data[i] = x.second.data[i];
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i];
            res.second.data[i] = x.second.data[i] + y.data[i];
        }
    }
}

template <typename T>
void rss_protocols::add(RSSTensor<T> &x, T y, RSSTensor<T> &res)
{
    int party_id = Party3PC::getInstance().party_id;
    res.assert_same_shape(x);

    if (party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i] + y;
            res.second.data[i] = x.second.data[i];
        }
    }
    else if (party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i];
            res.second.data[i] = x.second.data[i];
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i];
            res.second.data[i] = x.second.data[i] + y;
        }
    }
}

template <typename T>
void rss_protocols::sub(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res)
{
    res.assert_same_shape(x);
    res.assert_same_shape(y);

#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        res.first.data[i] = x.first.data[i] - y.first.data[i];
        res.second.data[i] = x.second.data[i] - y.second.data[i];
    }
}

template <typename T>
void rss_protocols::sub(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &res)
{
    int party_id = Party3PC::getInstance().party_id;
    res.assert_same_shape(x);
    res.assert_same_shape(y);

    if (party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i] - y.data[i];
            res.second.data[i] = x.second.data[i];
        }
    }
    else if (party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i];
            res.second.data[i] = x.second.data[i];
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i];
            res.second.data[i] = x.second.data[i] - y.data[i];
        }
    }
}

template <typename T>
void rss_protocols::sub(RSSTensor<T> &x, T y, RSSTensor<T> &res)
{
    int party_id = Party3PC::getInstance().party_id;
    res.assert_same_shape(x);

    if (party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i] - y;
            res.second.data[i] = x.second.data[i];
        }
    }
    else if (party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i];
            res.second.data[i] = x.second.data[i];
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i];
            res.second.data[i] = x.second.data[i] - y;
        }
    }
}

template <typename T>
void rss_protocols::sub(T x, RSSTensor<T> &y, RSSTensor<T> &res)
{
    int party_id = Party3PC::getInstance().party_id;
    res.assert_same_shape(y);

    if (party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < y.size(); i++)
        {
            res.first.data[i] = x - y.first.data[i];
            res.second.data[i] = -y.second.data[i];
        }
    }
    else if (party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < y.size(); i++)
        {
            res.first.data[i] = -y.first.data[i];
            res.second.data[i] = -y.second.data[i];
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < y.size(); i++)
        {
            res.first.data[i] = -y.first.data[i];
            res.second.data[i] = x - y.second.data[i];
        }
    }
}

template <typename T>
void rss_protocols::mulConst(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &res)
{
    res.assert_same_shape(x);
    res.assert_same_shape(y);

#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        res.first.data[i] = x.first.data[i] * y.data[i];
        res.second.data[i] = x.second.data[i] * y.data[i];
    }
}

template <typename T>
void rss_protocols::mulConst(RSSTensor<T> &x, T y, RSSTensor<T> &res)
{
    res.assert_same_shape(x);

#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        res.first.data[i] = x.first.data[i] * y;
        res.second.data[i] = x.second.data[i] * y;
    }
}

template <typename T>
void rss_protocols::mulConstAddBias(RSSTensor<T> &x, T y, T bias, RSSTensor<T> &res)
{
    int party_id = Party3PC::getInstance().party_id;
    res.assert_same_shape(x);

    if (party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i] * y + bias;
            res.second.data[i] = x.second.data[i] * y;
        }
    }
    else if (party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i] * y;
            res.second.data[i] = x.second.data[i] * y;
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i] * y;
            res.second.data[i] = x.second.data[i] * y + bias;
        }
    }
}

template <typename T>
void rss_protocols::mulConstAddBias(RSSTensor<T> &x, T y, RSSTensor<T> &bias, RSSTensor<T> &res)
{
    res.assert_same_shape(x);
    res.assert_same_shape(bias);

#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        res.first.data[i] = x.first.data[i] * y + bias.first.data[i];
        res.second.data[i] = x.second.data[i] * y + bias.second.data[i];
    }
}

template <typename T>
void rss_protocols::mulConstAddBias(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &bias, RSSTensor<T> &res)
{
    res.assert_same_shape(x);
    res.assert_same_shape(y);
    res.assert_same_shape(bias);

#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        res.first.data[i] = x.first.data[i] * y.data[i] + bias.first.data[i];
        res.second.data[i] = x.second.data[i] * y.data[i] + bias.second.data[i];
    }
}

template <typename T>
void rss_protocols::mulConstSubBias(RSSTensor<T> &x, T y, T bias, RSSTensor<T> &res)
{
    int party_id = Party3PC::getInstance().party_id;
    res.assert_same_shape(x);

    if (party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i] * y - bias;
            res.second.data[i] = x.second.data[i] * y;
        }
    }
    else if (party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i] * y;
            res.second.data[i] = x.second.data[i] * y;
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            res.first.data[i] = x.first.data[i] * y;
            res.second.data[i] = x.second.data[i] * y - bias;
        }
    }
}

template <typename T>
void rss_protocols::mulConstSubBias(RSSTensor<T> &x, T y, RSSTensor<T> &bias, RSSTensor<T> &res)
{
    res.assert_same_shape(x);
    res.assert_same_shape(bias);

#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        res.first.data[i] = x.first.data[i] * y - bias.first.data[i];
        res.second.data[i] = x.second.data[i] * y - bias.second.data[i];
    }
}

template <typename T>
void rss_protocols::mulConstSubBias(RSSTensor<T> &x, Tensor<T> &y, RSSTensor<T> &bias, RSSTensor<T> &res)
{
    res.assert_same_shape(x);
    res.assert_same_shape(bias);

#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        res.first.data[i] = x.first.data[i] * y.data[i] - bias.first.data[i];
        res.second.data[i] = x.second.data[i] * y.data[i] - bias.second.data[i];
    }
}

template <typename T>
void rss_protocols::mul(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, bool needTrunc, bool malicious)
{
    Party3PC &party = Party3PC::getInstance();
    res.assert_same_shape(x);
    res.assert_same_shape(y);
    uint32_t size = x.size();

    Tensor<T> tmp(x.shape);
#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        tmp.data[i] = x.first.data[i] * y.first.data[i] + x.first.data[i] * y.second.data[i] +
                      x.second.data[i] * y.first.data[i];
    }

    reshare(tmp, res);
    if (malicious)
    {
        RSSTensor<T> a(x.shape), b(y.shape), c(res.shape);
        a.zeros();
        b.zeros();
        c.zeros();

        uint32_t combined_size = 2 * size;
        RSSTensor<T> combined({combined_size});

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            combined.first.data[i] = a.first.data[i] - x.first.data[i];
            combined.second.data[i] = a.second.data[i] - x.second.data[i];
            combined.first.data[i + size] = b.first.data[i] - y.first.data[i];
            combined.second.data[i + size] = b.second.data[i] - y.second.data[i];
        }
        Tensor<T> rhoSigma({combined_size}), rho(x.shape), sigma(y.shape);
        restore(combined, rhoSigma, malicious);

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            rho.data[i] = rhoSigma.data[i];
            sigma.data[i] = rhoSigma.data[i + size];
        }

        RSSTensor<T> zero_res(x.shape);

        if (party.party_id == 0)
        {
#pragma omp parallel for
            for (int i = 0; i < size; i++)
            {
                zero_res.first.data[i] = res.first.data[i] - c.first.data[i] + a.first.data[i] * sigma.data[i] +
                                         rho.data[i] * b.first.data[i] - rho.data[i] * sigma.data[i];
                zero_res.second.data[i] = res.second.data[i] - c.second.data[i] + a.second.data[i] * sigma.data[i] +
                                          rho.data[i] * b.second.data[i];
            }
        }
        else if (party.party_id == 1)
        {
#pragma omp parallel for
            for (int i = 0; i < size; i++)
            {
                zero_res.first.data[i] = res.first.data[i] - c.first.data[i] + a.first.data[i] * sigma.data[i] +
                                         rho.data[i] * b.first.data[i];
                zero_res.second.data[i] = res.second.data[i] - c.second.data[i] + a.second.data[i] * sigma.data[i] +
                                          rho.data[i] * b.second.data[i];
            }
        }
        else
        {
#pragma omp parallel for
            for (int i = 0; i < size; i++)
            {
                zero_res.first.data[i] = res.first.data[i] - c.first.data[i] + a.first.data[i] * sigma.data[i] +
                                         rho.data[i] * b.first.data[i];
                zero_res.second.data[i] = res.second.data[i] - c.second.data[i] + a.second.data[i] * sigma.data[i] +
                                          rho.data[i] * b.second.data[i] - rho.data[i] * sigma.data[i];
            }
        }

        party.send_tensor_to(party.next_party_id, zero_res.first);
        Tensor<T> tmp_recv(x.shape);
        party.recv_tensor_from(party.pre_party_id, tmp_recv);

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            always_assert(zero_res.first.data[i] + zero_res.second.data[i] + tmp_recv.data[i] == 0);
        }
    }
    if (needTrunc)
    {
        truncate(res, malicious);
    }
}

template <typename T>
void rss_protocols::mul(RSSTensor<T> &x, std::pair<T, T> &y, RSSTensor<T> &res)
{
    res.assert_same_shape(x);
    uint32_t size = x.size();

    Tensor<T> tmp(x.shape);
#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        tmp.data[i] = x.first.data[i] * y.first + x.first.data[i] * y.second +
                      x.second.data[i] * y.first;
    }

    reshare(tmp, res);
}

template <typename T>
void rss_protocols::square(RSSTensor<T> &x, RSSTensor<T> &res, bool needTrunc, bool malicious)
{
    Party3PC &party = Party3PC::getInstance();
    res.assert_same_shape(x);
    uint32_t size = x.size();

    Tensor<T> tmp(x.shape);
#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        tmp.data[i] = x.first.data[i] * x.first.data[i] + x.first.data[i] * x.second.data[i] +
                      x.second.data[i] * x.first.data[i];
    }

    reshare(tmp, res);
    if (malicious)
    {
        RSSTensor<T> a(x.shape), c(res.shape); // c = a * a
        a.zeros();
        c.zeros();

        RSSTensor<T> rho_share({size});

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            rho_share.first.data[i] = a.first.data[i] - x.first.data[i];
            rho_share.second.data[i] = a.second.data[i] - x.second.data[i];
        }
        Tensor<T> rho({size});
        restore(rho_share, rho, malicious);

        RSSTensor<T> zero_res({size});

        if (party.party_id == 0)
        {
#pragma omp parallel for
            for (int i = 0; i < size; i++)
            {
                zero_res.first.data[i] = res.first.data[i] - c.first.data[i] + a.first.data[i] * rho.data[i] * 2 - rho.data[i] * rho.data[i];
                zero_res.second.data[i] = res.second.data[i] - c.second.data[i] + a.second.data[i] * rho.data[i] * 2;
            }
        }
        else if (party.party_id == 1)
        {
#pragma omp parallel for
            for (int i = 0; i < size; i++)
            {
                zero_res.first.data[i] = res.first.data[i] - c.first.data[i] + a.first.data[i] * rho.data[i] * 2;
                zero_res.second.data[i] = res.second.data[i] - c.second.data[i] + a.second.data[i] * rho.data[i] * 2;
            }
        }
        else
        {
#pragma omp parallel for
            for (int i = 0; i < size; i++)
            {
                zero_res.first.data[i] = res.first.data[i] - c.first.data[i] + a.first.data[i] * rho.data[i] * 2;
                zero_res.second.data[i] = res.second.data[i] - c.second.data[i] + a.second.data[i] * rho.data[i] * 2 -
                                          rho.data[i] * rho.data[i];
            }
        }

        party.send_tensor_to(party.next_party_id, zero_res.first);
        Tensor<T> tmp_recv({size});
        party.recv_tensor_from(party.pre_party_id, tmp_recv);

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            always_assert(zero_res.first.data[i] + zero_res.second.data[i] + tmp_recv.data[i] == 0);
        }
    }
    if (needTrunc)
    {
        truncate(res, malicious);
    }
}

template <typename T>
void rss_protocols::matMul(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, bool needTrunc, bool malicious)
{
    Party3PC &party = Party3PC::getInstance();
    int threads_num = omp_get_max_threads();
    omp_set_num_threads(64);

    uint32_t size = res.size(), x_size = x.size(), y_size = y.size();
    Tensor<T> tmp(res.shape);

    const uint32_t x_shape_size = x.shape.size();
    const uint32_t y_shape_size = y.shape.size();
    const uint32_t z_shape_size = res.shape.size();
    always_assert(x_shape_size >= 2 && y_shape_size >= 2 && z_shape_size >= 2);
    always_assert(x.shape[x_shape_size - 1] == y.shape[y_shape_size - 2]);
    always_assert(res.shape[z_shape_size - 2] == x.shape[x_shape_size - 2]);
    always_assert(res.shape[z_shape_size - 1] == y.shape[y_shape_size - 1]);

    uint32_t row, common_dim, col, broadcast_size, common_size;
    bool is_b_broadcast;

    compute_matmul_info(x.shape, y.shape, res.shape, row, common_dim, col, broadcast_size, common_size,
                        is_b_broadcast);

    utils::RSSMatMul(x, y, tmp, broadcast_size, common_size, row, common_dim, col, is_b_broadcast);
    reshare(tmp, res);

    if (malicious)
    {
        RSSTensor<T> a(x.shape), b(y.shape), c(res.shape);
        a.zeros();
        b.zeros();
        c.zeros();

        uint32_t combined_size = x_size + y_size;
        RSSTensor<T> combined({combined_size});

#pragma omp parallel for
        for (int i = 0; i < x_size; i++)
        {
            combined.first.data[i] = a.first.data[i] - x.first.data[i];
            combined.second.data[i] = a.second.data[i] - x.second.data[i];
        }
#pragma omp parallel for
        for (int i = 0; i < y_size; i++)
        {
            combined.first.data[i + x_size] = b.first.data[i] - y.first.data[i];
            combined.second.data[i + x_size] = b.second.data[i] - y.second.data[i];
        }

        Tensor<T> rhoSigma({combined_size}), rho(x.shape), sigma(y.shape);
        restore(combined, rhoSigma, malicious);

#pragma omp parallel for
        for (int i = 0; i < x_size; i++)
        {
            rho.data[i] = rhoSigma.data[i];
        }
        for (int i = 0; i < y_size; i++)
        {
            sigma.data[i] = rhoSigma.data[i + x_size];
        }

        RSSTensor<T> zero_res(res.shape), af(res.shape), eb(res.shape);
        Tensor<T> ef(res.shape);

        utils::RSSMatMul(a, sigma, af, broadcast_size, common_size, row, common_dim, col, is_b_broadcast);
        utils::RSSMatMul(rho, b, eb, broadcast_size, common_size, row, common_dim, col, is_b_broadcast);
        Tensor<T>::matmul(ef, rho, sigma, broadcast_size, common_size, row, common_dim, col, is_b_broadcast);

        if (party.party_id == 0)
        {
#pragma omp parallel for
            for (int i = 0; i < size; i++)
            {
                zero_res.first.data[i] =
                    res.first.data[i] - c.first.data[i] + af.first.data[i] + eb.first.data[i] - ef.data[i];
                zero_res.second.data[i] = res.second.data[i] - c.second.data[i] + af.second.data[i] + eb.second.data[i];
            }
        }
        else if (party.party_id == 1)
        {
#pragma omp parallel for
            for (int i = 0; i < size; i++)
            {
                zero_res.first.data[i] = res.first.data[i] - c.first.data[i] + af.first.data[i] + eb.first.data[i];
                zero_res.second.data[i] = res.second.data[i] - c.second.data[i] + af.second.data[i] + eb.second.data[i];
            }
        }
        else
        {
#pragma omp parallel for
            for (int i = 0; i < size; i++)
            {
                zero_res.first.data[i] = res.first.data[i] - c.first.data[i] + af.first.data[i] + eb.first.data[i];
                zero_res.second.data[i] =
                    res.second.data[i] - c.second.data[i] + af.second.data[i] + eb.second.data[i] - ef.data[i];
            }
        }

        party.send_tensor_to(party.next_party_id, zero_res.first);
        Tensor<T> tmp_recv(res.shape);
        party.recv_tensor_from(party.pre_party_id, tmp_recv);

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            always_assert(zero_res.first.data[i] + zero_res.second.data[i] + tmp_recv.data[i] == 0);
        }
    }
    omp_set_num_threads(threads_num);
    if (needTrunc)
    {
        truncate(res, malicious);
    }
}

template <typename T>
void rss_protocols::truncate(RSSTensor<T> &x, RSSTensor<T> &res, size_t scale, bool malicious)
{
    Party3PC &party = Party3PC::getInstance();
    RSSTensor<T> r(x.shape), r_t(x.shape);
    uint32_t size = x.size();
    r.zeros();
    r_t.zeros();

    sub(x, r, x);
    Tensor<T> x_shift(x.shape);
    restore(x, x_shift, malicious);

    if (party.party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            if constexpr (std::is_same_v<T, uint32_t>)
            {
                res.first.data[i] =
                    r_t.first.data[i] + (T)((double)((int32_t)x_shift.data[i]) / (double)((int32_t)scale));
            }
            else
            {
                res.first.data[i] =
                    r_t.first.data[i] + (T)((double)((int64_t)x_shift.data[i]) / (double)((int64_t)scale));
            }

            res.second.data[i] = r_t.second.data[i];
        }
    }
    else if (party.party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            res.first.data[i] = r_t.first.data[i];
            res.second.data[i] = r_t.second.data[i];
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            res.first.data[i] = r_t.first.data[i];
            if constexpr (std::is_same_v<T, uint32_t>)
            {
                res.second.data[i] =
                    r_t.second.data[i] + (T)((double)((int32_t)x_shift.data[i]) / (double)((int32_t)scale));
            }
            else
            {
                res.second.data[i] =
                    r_t.second.data[i] + (T)((double)((int64_t)x_shift.data[i]) / (double)((int64_t)scale));
            }
        }
    }
}

template <typename T>
void rss_protocols::truncate(RSSTensor<T> &x, size_t scale, bool malicious)
{
    truncate(x, x, scale, malicious);
}

template <typename T>
void rss_protocols::truncate(RSSTensor<T> &x, bool malicious)
{
    truncate(x, 1 << x.float_scale_bit, malicious);
}

template <typename T>
void rss_protocols::checkZero(RSSTensor<T> &x)
{
    RSSTensor<T> r(x.shape), xr(x.shape);
    Tensor<T> xr_open(x.shape);

    r.zeros(); // it should be random
    mul(x, r, xr);
    restore(xr, xr_open, true);

    xr_open.assert_zero();
}

template <typename T>
void rss_protocols::macCheck(const RSSTensor<T> &x, const RSSTensor<T> &mx, const std::pair<T, T> &mac_key)
{
#if (IS_MALICIOUS)
    Party3PC &party = Party3PC::getInstance();
    RSSTensor<T> r({1}), mr({1});
    r.zeros(); // it should be random
    mul(r, mac_key, mr);
    RSSTensor<T> ro_share(x.shape);
    ro_share.zeros(); // it should be random
    Tensor<T> ro(x.shape);
    restore(ro_share, ro, true);
    RSSTensor<T> v({1}), w({1});
    v.first.data[0] = r.first.data[0];
    v.second.data[0] = r.second.data[0];

    w.first.data[0] = mr.first.data[0];
    w.second.data[0] = mr.second.data[0];

    for (int i = 0; i < x.size(); i++)
    {
        v.first.data[i] += x.first.data[i] * ro.data[i];
        v.second.data[i] += x.second.data[i] * ro.data[i];
        w.first.data[i] += mx.first.data[i] * ro.data[i];
        w.second.data[i] += mx.second.data[i] * ro.data[i];
    }
    Tensor<T> v_real({1});
    restore(v, v_real, true);
    RSSTensor<T> delta({1});
    mulConstSubBias(mac_key, v_real, w, delta);
    checkZero(delta);
#endif
}

template <typename T>
void rss_protocols::macCheck(RSSTensor<T> &x, RSSTensor<T> &mx, RSSTensor<T> &mac_key)
{
#if (IS_MALICIOUS)
    RSSTensor<T> r(x.shape), mr(mx.shape);
    r.zeros(); // it should be random
    mul(r, mac_key, mr);
    RSSTensor<T> ro_share(x.shape);
    ro_share.zeros(); // it should be random
    Tensor<T> ro(x.shape);
    restore(ro_share, ro, true);
    RSSTensor<T> v(x.shape), w(x.shape);
    mulConstAddBias(x, ro, r, v);
    mulConstAddBias(mx, ro, mr, w);
    Tensor<T> v_real(x.shape);
    restore(v, v_real, true);
    RSSTensor<T> delta(x.shape);
    mulConstSubBias(mac_key, v_real, w, delta);
    checkZero(delta);
#endif
}

template <typename T>
void rss_protocols::macCheckSimulate(uint32_t size)
{
#if (IS_MALICIOUS)
    if (size == 0)
    {
        return;
    }
    RSSTensor<T> x({size}), mx({size}), mac_key({size});
    x.zeros();
    mx.zeros();
    mac_key.zeros();

    macCheck(x, mx, mac_key);
    MAC_SIZE = 0;
#endif
}

template <typename T>
void rss_protocols::utils::privateCompare(Tensor<T> &x_with_pre, Tensor<T> &x_with_next, Tensor<T> &res_with_pre,
                                          Tensor<T> &res_with_next, Parameters<T> &parameter)
{
    Party3PC &party = Party3PC::getInstance();
    // r_bit_with_pre and r_bit_with_next are provided by parameter
    uint32_t size = x_with_pre.size();
    uint32_t bit_length = 8 * sizeof(T);
    uint32_t double_bit_length = 2 * bit_length;
    uint32_t long_size = size * (bit_length - 1); // The actual input is \ell - 1 bit and only need to calculate \ell - 1 bit

    Tensor<uint8_t> c_with_pre({long_size}), c_with_next({long_size});
    T w_with_pre, w_with_next, x_bit_with_pre, x_bit_with_next, w_sum_with_pre, w_sum_with_next;
    T r_with_pre_bit, r_with_next_bit;

    int index;
#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        w_sum_with_pre = 0;
        w_sum_with_next = 0;

        for (int j = 1; j < bit_length; j++)
        {
            x_bit_with_pre = (x_with_pre.data[i] >> (bit_length - 1 - j)) & 1;
            x_bit_with_next = (x_with_next.data[i] >> (bit_length - 1 - j)) & 1;

            r_with_pre_bit = parameter.pc_cmp.r_with_pre_bits.data[j];
            r_with_next_bit = parameter.pc_cmp.r_with_next_bits.data[j];

            if (x_bit_with_pre == 0)
            {
                w_with_pre = r_with_pre_bit;
            }
            else
            {
                w_with_pre = -r_with_pre_bit;
            }

            if (x_bit_with_next == 0)
            {
                w_with_next = r_with_next_bit;
            }
            else
            {
                w_with_next = 1 - r_with_next_bit;
            }

            index = i * (bit_length - 1) + j - 1;
            c_with_pre.data[index] = (uint8_t)(r_with_pre_bit + w_sum_with_pre + parameter.pc_cmp.round1_r);
            c_with_next.data[index] = (uint8_t)(r_with_next_bit - x_bit_with_next + w_sum_with_next + 1 + parameter.pc_cmp.round1_r);

            w_sum_with_pre += w_with_pre;
            w_sum_with_next += w_with_next;
        }
    }
    Tensor<uint8_t> c_with_pre_tmp({long_size}), c_with_next_tmp({long_size});

    if (party.party_id == 2)
    {
        party.send_tensor_to(party.next_party_id, c_with_next);
        party.recv_tensor_from(party.next_party_id, c_with_next_tmp);

        party.send_tensor_to(party.pre_party_id, c_with_pre);
        party.recv_tensor_from(party.pre_party_id, c_with_pre_tmp);
    }
    else
    {
        party.send_tensor_to(party.pre_party_id, c_with_pre);
        party.recv_tensor_from(party.pre_party_id, c_with_pre_tmp);

        party.send_tensor_to(party.next_party_id, c_with_next);
        party.recv_tensor_from(party.next_party_id, c_with_next_tmp);
    }

#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        res_with_pre.data[i] = 0;
        res_with_next.data[i] = 0;
        for (int j = 0; j < bit_length - 1; j++)
        {
            index = i * (bit_length - 1) + j;
            res_with_pre.data[i] += parameter.pc_cmp.round1_real_table_with_pre.data[(c_with_pre.data[index] + c_with_pre_tmp.data[index]) % double_bit_length];
            res_with_next.data[i] += parameter.pc_cmp.round1_real_table_with_next.data[(c_with_next.data[index] + c_with_next_tmp.data[index]) % double_bit_length];
        }
    }
}

template <typename T>
void rss_protocols::pc_msb(RSSTensor<T> &x, Tensor<T> &res_with_pre, Tensor<T> &res_with_next,
                           Parameters<T> &parameter, const uint32_t size, bool malicious)
{
}

template <typename T>
void rss_protocols::nonNegative(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameter, bool isFloat, bool malicious)
{
}

template <typename T>
void rss_protocols::greaterEqual(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, Parameters<T> &parameter, bool isFloat, bool malicious)
{
    RSSTensor<T> z(x.shape);
    sub(x, y, z);
    nonNegative(z, res, parameter, isFloat, malicious);
}

template <typename T>
void rss_protocols::b2a(RSSTensor<T> &x, RSSTensor<T> &res, bool malicious)
{
    Party3PC &party = Party3PC::getInstance();
    uint32_t size = x.size();
    RSSTensor<T> r(x.shape), a(x.shape);
    r.zeros();

    for (int i = 0; i < size; i++)
    {
        a.first.data[i] = x.first.data[i] ^ r.first.data[i];
        a.second.data[i] = x.second.data[i] ^ r.second.data[i];
    }

    Tensor<T> a_res(x.shape);
    restore_bit(a, a_res, malicious);

    if(party.party_id == 0)
    {
        for (int i = 0; i < size; i++)
        {
            res.first.data[i] = a_res.data[i] + r.first.data[i] - 2 * a_res.data[i] * r.first.data[i];
            res.second.data[i] = r.second.data[i] - 2 * a_res.data[i] * r.second.data[i];
        }  
    }  
    else if(party.party_id == 1)
    {
        for (int i = 0; i < size; i++)
        {
            res.first.data[i] = r.first.data[i] - 2 * a_res.data[i] * r.first.data[i];
            res.second.data[i] = r.second.data[i] - 2 * a_res.data[i] * r.second.data[i];
        }  
    }
    else
    {
        for (int i = 0; i < size; i++)
        {
            res.first.data[i] = r.first.data[i] - 2 * a_res.data[i] * r.first.data[i];
            res.second.data[i] = a_res.data[i] + r.second.data[i] - 2 * a_res.data[i] * r.second.data[i];
        }  
    }
}

template <typename T>
void rss_protocols::equal_judge(Tensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameter, bool malicious)
{
    Party3PC &party = Party3PC::getInstance();
    RSSTensor<uint8_t> table_1 = parameter.equal_param.table_1;
    uint32_t size = x.size();
    uint8_t d_size = sizeof(T);

    res.zeros();

#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < d_size; j++)
        {
            uint8_t tmp_v = (x.data[i] >> (8 * j)) & 0xFF;
            uint32_t index = 8 * sizeof(T) * j + tmp_v / 8;

            res.first.data[i] += (table_1.first.data[index] >> (7 - tmp_v % 8) & 1) << j;
            res.second.data[i] += (table_1.second.data[index] >> (7 - tmp_v % 8) & 1) << j;
        }
    }

    if(d_size != 1)
    {
        RSSTensor<uint8_t> table_2 = parameter.equal_param.table_2;
        uint8_t r_21 = parameter.equal_param.r_21;
        uint8_t r_22 = parameter.equal_param.r_22;

        RSSTensor<uint8_t> v_hat(x.shape);
        Tensor<uint8_t> v_hat_res(x.shape);

        if(party.party_id == 0)
        {
            uint8_t mask = ((1 << d_size) - 1) ^ r_21;

#pragma omp parallel for
            for (int i = 0; i < size; i++)
            {
                v_hat.first.data[i] = res.first.data[i] ^ mask;
                v_hat.second.data[i] = res.second.data[i] ^ r_22;
            }
        }
        else if(party.party_id == 1)
        {
#pragma omp parallel for
            for (int i = 0; i < size; i++)
            {
                v_hat.first.data[i] = res.first.data[i] ^ r_21;
                v_hat.second.data[i] = res.second.data[i] ^ r_22;
            }    
        }
        else
        {
            uint8_t mask = ((1 << d_size) - 1) ^ r_22;

#pragma omp parallel for
            for (int i = 0; i < size; i++)
            {
                v_hat.first.data[i] = res.first.data[i] ^ r_21;
                v_hat.second.data[i] = res.second.data[i] ^ mask;
            }
        }
        restore_bit(v_hat, v_hat_res, malicious);    
            
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            int index = v_hat_res.data[i] / 8, delta = 7 - v_hat_res.data[i] % 8;
            res.first.data[i] = table_2.first.data[index] >> delta & 1;
            res.second.data[i] = table_2.second.data[index] >> delta & 1;
        }   
    }
}

template <typename T>
void rss_protocols::table_Equal(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, 
                                Parameters<T> &parameter, bool is_b2a, bool isFloat, bool malicious)
{
    Party3PC &party = Party3PC::getInstance();
    T r_11 = parameter.equal_param.r_11;
    T r_12 = parameter.equal_param.r_12;

    // 盲化输入
    RSSTensor<T> tmp(x.shape);
#pragma omp parallel for
    for (int i = 0; i < x.size(); i++)
    {
        tmp.first.data[i] = x.first.data[i] - y.first.data[i] + r_11;
        tmp.second.data[i] = x.second.data[i] - y.second.data[i] + r_12;
    }

    Tensor<T> tmp_res(x.shape);
    restore(tmp, tmp_res);

    // 等于判断
    equal_judge(tmp_res, res, parameter);
    
    std::cout << res.first.data[1] << std::endl;

    // b2a
    if (is_b2a) 
        b2a(res, res, malicious);

    std::cout << res.first.data[1] << std::endl;

    if (isFloat)
        mulConst(res, (T)1 << kFloat_Precision<T>, res);
}

template <typename T>
void rss_protocols::greater_judge(Tensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameter, bool malicious)
{
    Party3PC &party = Party3PC::getInstance();

    uint32_t size = x.size(); // 元素个数
    uint8_t d_size = sizeof(T); // data_size

    RSSTensor<uint8_t> table_1 = parameter.greater_param.table_1;
    RSSTensor<uint8_t> table_2 = parameter.greater_param.table_2;

    std::vector<uint32_t> bc_shape = {size, sizeof(T)};
    RSSTensor<uint8_t> b(bc_shape), c(bc_shape);
    b.zeros();

    // 生成b和c
#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        T base = i*d_size;
        for (int j = 0; j < d_size; j++)
        {
            uint8_t tmp_v = (x.data[i] >> (8 * j)) & 0xFF;
            T index = 8 * sizeof(T) * j + tmp_v / 8;
            T j_index = base + j; 

            c.first.data[j_index] = (table_1.first.data[index] >> (7 - tmp_v % 8) & 1); // 判断相等
            c.second.data[j_index] = (table_1.second.data[index] >> (7 - tmp_v % 8) & 1);
            
            b.first.data[j_index] = table_1.first.data[32 * j];
            b.second.data[j_index] = table_1.second.data[32 * j];
            for (int k = 32 * j + 1; k < index; k++)
            {
                b.first.data[j_index] ^= table_1.first.data[k];
                b.second.data[j_index] ^= table_1.first.data[k];
                // 0-(index-1)并行化计算异或值，index循环计算异或值    
            }
            b.first.data[j_index] = __builtin_parity(b.first.data[j_index]);
            b.second.data[j_index] = __builtin_parity(b.second.data[j_index]);

            for (int k = 0; k < tmp_v % 8; k++)
            {
                b.first.data[j_index] ^= (table_1.first.data[index] >> (7 - k) & 1);
                b.second.data[j_index] ^= (table_1.second.data[index] >> (7 - k) & 1);
            }
        }   
    }

    uint8_t r_21 = parameter.greater_param.r_21;
    uint8_t r_22 = parameter.greater_param.r_22;
    std::vector<uint32_t> v_shape = {size, d_size};
    RSSTensor<uint8_t> v(v_shape);

    if(party.party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            T base = i * d_size;
            for (int j = 0; j < d_size; j++)
            {
                T j_index = base + j;
                v.first.data[j_index] = b.first.data[j_index];
                v.second.data[j_index] = b.second.data[j_index];

                for (int k = j + 1; k < d_size; k++)
                {
                    v.first.data[j_index] += c.first.data[base + k] << (k - j);
                    v.second.data[j_index] += c.second.data[base + k] << (k - j);
                }
                    
                v.first.data[j_index] = v.first.data[j_index] ^ ((1 << d_size-j) - 1) ^ r_21;
                v.second.data[j_index] = v.second.data[j_index] ^ r_22;
            } 
        }
    }
    else if (party.party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            T base = i * d_size;
            for (int j = 0; j < d_size; j++)
            {
                T j_index = base + j;
                v.first.data[j_index] = b.first.data[j_index];
                v.second.data[j_index] = b.second.data[j_index];

                for (int k = j + 1; k < d_size; k++)
                {
                    v.first.data[j_index] += c.first.data[base + k] << (k - j);
                    v.second.data[j_index] += c.second.data[base + k] << (k - j);
                }
                    
                v.first.data[j_index] = v.first.data[j_index] ^ r_21;
                v.second.data[j_index] = v.second.data[j_index] ^ r_22;
            } 
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            T base = i * d_size;
            for (int j = 0; j < d_size; j++)
            {
                T j_index = base + j;
                v.first.data[j_index] = b.first.data[j_index];
                v.second.data[j_index] = b.second.data[j_index];

                for (int k = j + 1; k < d_size; k++)
                {
                    v.first.data[j_index] += c.first.data[base + k] << (k - j);
                    v.second.data[j_index] += c.second.data[base + k] << (k - j);
                }
                    
                v.first.data[j_index] = v.first.data[j_index] ^ r_21;
                v.second.data[j_index] = v.second.data[j_index]  ^ ((1 << d_size-j) - 1) ^ r_22;
            } 
        }
    }

    Tensor<uint8_t> v_hat({size, (uint32_t)d_size});
    restore_bit(v, v_hat, malicious);

#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        T base = i * d_size;
        uint8_t val = table_2.second.data[0] >> 7 & 1;
        for (int j = 0; j < d_size; j++)
        {
            int j_index = base + j;
            int index = v_hat.data[j_index] / 8, delta = 7 - v_hat.data[j_index] % 8;
            v.first.data[j_index] = table_2.first.data[index] >> delta & 1;
            v.second.data[j_index] = table_2.second.data[index] >> delta & 1;
        }
    }

#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        res.first.data[i] = 0;
        res.second.data[i] = 0;
        T base = i * d_size;
        for (int j = 0; j < d_size; j++)
        {
            int j_index = base + j;
            res.first.data[i] ^= v.first.data[j_index];
            res.second.data[i] ^= v.second.data[j_index];
        }
    }
}

template <typename T>
void rss_protocols::table_greaterEqual(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, 
                                       Parameters<T> &parameter, bool is_b2a, bool isFloat, bool malicious)
{
    Party3PC &party = Party3PC::getInstance();
    uint32_t size = x.size();
    uint8_t llength = 8 * sizeof(T) - 1;
    
    T r_11 = parameter.greater_param.r_11;
    T r_12 = parameter.greater_param.r_12;
    T r_1 = parameter.greater_param.r_1;
    T r_2 = parameter.greater_param.r_2;
    uint8_t msb_2_l_r_1 = parameter.greater_param.msb_2_l_r_1;
    uint8_t msb_2_l_r_2 = parameter.greater_param.msb_2_l_r_2;

    RSSTensor<T> x_hat(x.shape);
#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        x_hat.first.data[i] = x.first.data[i] - y.first.data[i] + r_1;
        x_hat.second.data[i] = x.second.data[i] - y.second.data[i] + r_2;
    }
    
    Tensor<T> x_hat_res(x.shape), y_0(x.shape);
    restore(x_hat, x_hat_res);

    T mask = 1 << llength;
#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        y_0.data[i] = x_hat_res.data[i] % mask;
    }

    RSSTensor<T> greater_res(x.shape);
    greater_judge(y_0, greater_res, parameter);

    if (party.party_id == 0)
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            res.first.data[i] = (x_hat_res.data[i] >> llength & 1) ^ msb_2_l_r_1 ^ greater_res.first.data[i];
            res.second.data[i] =  msb_2_l_r_2 ^ greater_res.second.data[i];
        }
    }
    else if (party.party_id == 1)
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            res.first.data[i] =  msb_2_l_r_1 ^ greater_res.first.data[i];
            res.second.data[i] =  msb_2_l_r_2 ^ greater_res.second.data[i];
        }
    }
    else 
    {
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            res.first.data[i] = msb_2_l_r_1 ^ greater_res.first.data[i];
            res.second.data[i] = (x_hat_res.data[i] >> llength & 1) ^ msb_2_l_r_2 ^ greater_res.second.data[i];
        }
    }

    sub((T)1, res, res);

    std::cout << res.first.data[1] << std::endl;
    // b2a
    if (is_b2a) 
        b2a(res, res, malicious);

    std::cout << res.first.data[1] << std::endl;

    if(isFloat)
        mulConst(res, (T)1 << kFloat_Precision<T>, res);
} 

template <typename T>
void rss_protocols::select(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, Parameters<T> &parameter, bool malicious)
{
}

template <typename T>
void rss_protocols::select(RSSTensor<T> &x, RSSTensor<T> &y, RSSTensor<T> &res, uint32_t y_num, Parameters<T> &parameter, bool malicious)
{
}

template <typename T>
void rss_protocols::lut(RSSTensor<T> &x, RSSTensor<T> &res, LUT_Param<T> &parameter, bool malicious)
{
}

template <typename T>
void rss_protocols::lut(RSSTensor<T> &x, RSSTensor<T> &res1, RSSTensor<T> &res2, LUT_Param<T> &parameter1, LUT_Param<T> &parameter2, bool malicious)
{
}

template <typename T>
void rss_protocols::utils::getk(RSSTensor<T> &x, RSSTensor<T> &k, Parameters<T> &parameters, bool malicious)
{
}

template <typename T>
void rss_protocols::inv(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious)
{
}

template <typename T>
void rss_protocols::rsqrt(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious)
{
}

template <typename T>
void rss_protocols::utils::gelu_same_scale(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious)
{
}

template <typename T>
void rss_protocols::gelu(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameters, bool malicious)
{
}

template <typename T>
void rss_protocols::max_last_dim(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameter, bool malicious)
{
}

template <typename T>
void rss_protocols::neg_exp(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameter, bool malicious)
{
}

template <typename T>
void rss_protocols::softmax_forward(RSSTensor<T> &x, RSSTensor<T> &res, Parameters<T> &parameter, bool malicious)
{
}

template <typename T, typename U>
void rss_protocols::downcast(RSSTensor<T> &x, RSSTensor<U> &res)
{
}

template <typename U, typename T>
void rss_protocols::upcast(RSSTensor<U> &x, RSSTensor<T> &res, int party_id, bool malicious)
{
}
