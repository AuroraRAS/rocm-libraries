/*******************************************************************************
 * Copyright Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *******************************************************************************/

#include <gtest/gtest.h>

#include <msgpack.hpp>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionTaskPredicates.hpp>
#include <Tensile/PlaceholderLibrary.hpp>
#include <Tensile/Serialization/Predicates.hpp>
#include <Tensile/msgpack/MessagePack.hpp>

namespace
{
    using namespace TensileLite;
}

TEST(Gfx90cSupport, ArchitectureConversionHandlesTargetIds)
{
    EXPECT_EQ(AMDGPU::toProcessor("gfx90c"), AMDGPU::Processor::gfx90c);
    EXPECT_EQ(AMDGPU::toProcessor("gfx90c:xnack+"), AMDGPU::Processor::gfx90c);
    EXPECT_EQ(AMDGPU::toString(AMDGPU::Processor::gfx90c), "gfx90c");
}

TEST(Gfx90cSupport, ProcessorDeserializesFromMessagePack)
{
    msgpack::sbuffer buffer;
    msgpack::pack(buffer, std::string("gfx90c"));
    auto object = msgpack::unpack(buffer.data(), buffer.size());

    AMDGPU::Processor               processor = AMDGPU::Processor::gfx000;
    Serialization::MessagePackInput input(object.get());
    input.input(processor);

    EXPECT_TRUE(input.error.empty());
    EXPECT_EQ(processor, AMDGPU::Processor::gfx90c);
}

TEST(Gfx90cSupport, LazyLoadingRoutesTargetIdsToGfx90cShard)
{
    EXPECT_EQ(LazyLoadingInitFromArchitecture("gfx90c"), LazyLoadingInit::gfx90c);
    EXPECT_EQ(LazyLoadingInitFromArchitecture("gfx90c:xnack+"), LazyLoadingInit::gfx90c);
    EXPECT_EQ(LazyLoadingInitFromArchitecture("gfx90c:xnack-"), LazyLoadingInit::gfx90c);
    EXPECT_EQ(RegexPattern(LazyLoadingInit::gfx90c), "TensileLibrary_*_gfx90c");
}
