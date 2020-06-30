#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////

// Intrinsics for POPCNT/PEXT/PDEP/LSB/MSB
// If disabled, a generic implementation is used

#define USE_BMI2
#define USE_LSB
#define USE_POPCNT

// Lookup tables for various Square -> Bitboard calculations

#define USE_SQUARE_BB	// 1 << sq_as_int
//#define USE_BETWEEN_BB	// line_between(Square, Square)

#define USE_KNIGHT_BB	// Knight attacks
#define USE_KING_BB		// King attacks
#define USE_BISHOP_BB	// Bishop attacks w/o occupancy
#define USE_ROOK_BB		// Rook attacks w/o occupancy
#define USE_QUEEN_BB	// Queen attacks w/o occupancy

// Sliding-piece attack generation
// (pick one only, if PEXT/PDEP ensure you have BMI2 extensions enabled)

//#define USE_KOGGE
//#define USE_FANCY
#define USE_PEXT
//#define USE_PDEP

////////////////////////////////////////////////////////////////////////////////////////////////////

#include <array>
#include <charconv>
#include <cstdint>
#include <type_traits>

#define FMT_HEADER_ONLY
#include "fmt/format.h"

#if defined(USE_BMI2)
#	include <immintrin.h>
#endif

#if defined(USE_LSB)
#	if defined(_MSC_VER)
#		include <intrin.h>
#	endif
#endif

#if defined(USE_POPCNT)
#	if defined(_MSC_VER) || defined(__INTEL_COMPILER)
#		include <nmmintrin.h>
#	elif defined(__GNUC__)
#	endif
#endif

#define ASSERT(cond) ((void) 0)

//
// Intrinsics
//

#if defined(USE_BMI2)
constexpr bool HasBMI2 = true;

inline std::uint64_t pext(std::uint64_t x, std::uint64_t mask)
{
	return _pext_u64(x, mask);
}

inline std::uint64_t pdep(std::uint64_t x, std::uint64_t mask)
{
	return _pdep_u64(x, mask);
}
#else
constexpr bool HasBMI2 = false;

constexpr std::uint64_t pdep(const std::uint64_t x, std::uint64_t mask)
{
	std::uint64_t res = 0;

	for (std::uint64_t bb = 1; mask; bb += bb)
	{
		if (x & bb)
			res |= mask & (-mask);

		mask &= (mask - 1);
	}

	return res;
}

constexpr std::uint64_t pext(const std::uint64_t x, std::uint64_t mask)
{
	std::uint64_t res = 0;

	for (std::uint64_t bb = 1; mask; bb += bb)
	{
		if (x & mask & -mask)
			res |= bb;

		mask &= (mask - 1);
	}

	return res;
}
#endif

#if defined(USE_LSB)
constexpr bool HasLsbIntrinsics = true;

#	if defined(__GNUC__)
inline unsigned lsb(const std::uint64_t x)
{
	ASSERT(x != 0);
	return __builtin_ctzll(x);
}

inline unsigned msb(const std::uint64_t x)
{
	ASSERT(x != 0);
	return __builtin_clzll(x) ^ 63u;
}
#	elif defined(_MSC_VER)
inline unsigned lsb(const std::uint64_t x)
{
	ASSERT(x != 0);

	unsigned long result;
	_BitScanForward64(&result, x);
	return static_cast<unsigned>(result);
}

inline unsigned msb(const std::uint64_t x)
{
	ASSERT(x != 0);

	unsigned long result;
	_BitScanReverse64(&result, x);
	static_cast<unsigned>(result);
}
#	else
#		error "LSB intrinsics not supported"
#	endif
#else
constexpr bool HasLsbIntrinsics = false;

constexpr unsigned lsb(std::uint64_t x)
{
	constexpr std::uint64_t lsb_magic_64 = 0x021937ec54bad1e7;
	constexpr unsigned lsb_index_64[] = {
		0,  1,	2,  7,  3, 30,  8,  52,
		4,  13, 31, 38, 9,  16, 59, 53,
		5,  50, 36, 14, 34, 32, 45, 39,
		27, 10, 47, 17, 60, 41, 54, 20,
		63, 6,  29, 51, 12, 37, 15, 58,
		49, 35, 33, 44, 26, 46, 40, 19,
		62, 28, 11, 57, 48, 43, 25, 18,
		61, 56, 42, 24, 55, 23, 22, 21
	};

	return lsb_index_64[((x & (~x + 1)) * lsb_magic_64) >> 58u];
}

constexpr unsigned msb(std::uint64_t x)
{
	constexpr std::uint64_t msb_magic_64 = 0x03f79d71b4cb0a89;
	constexpr unsigned msb_index_64[] = {
		0,  47, 1,  56, 48, 27, 2,  60,
		57, 49, 41, 37, 28, 16, 3,  61,
		54, 58, 35, 52, 50, 42, 21, 44,
		38, 32, 29, 23, 17, 11, 4,  62,
		46, 55, 26, 59, 40, 36, 15, 53,
		34, 51, 20, 43, 31, 22, 10, 45,
		25, 39, 14, 33, 19, 30, 9,  24,
		13, 18, 8,  12, 7,  6,  5,  63
	};

	x |= x >> 1u;
	x |= x >> 2u;
	x |= x >> 4u;
	x |= x >> 8u;
	x |= x >> 16u;
	x |= x >> 32u;

	return msb_index_64[(x * msb_magic_64) >> 58u];
}
#endif

constexpr unsigned popcount_generic(std::uint64_t x)
{
	constexpr std::uint64_t m1	= 0x5555555555555555ull;
	constexpr std::uint64_t m2	= 0x3333333333333333ull;
	constexpr std::uint64_t m4	= 0x0F0F0F0F0F0F0F0Full;
	constexpr std::uint64_t h01 = 0x0101010101010101ull;

	x -= (x >> 1u) & m1;
	x = (x & m2) + ((x >> 2u) & m2);
	x = (x + (x >> 4u)) & m4;

	return (x * h01) >> 56u;
}

#if defined(USE_POPCNT)
constexpr bool HasPopcntIntrinsics = true;

#	if defined(_MSC_VER) || defined(__INTEL_COMPILER)
inline unsigned popcount(const std::uint64_t x)
{
	return _mm_popcnt_u64(x);
}
#	elif defined(__GNUC__)
inline unsigned popcount(const std::uint64_t x)
{
	return __builtin_popcountll(x);
}
#	else
#		error "POPCNT intrinsics not supported"
#	endif
#else
constexpr bool HasPopcntIntrinsics = false;
constexpr auto popcount = popcount_generic;
#endif

//
// Misc. functions
//

template <typename Enum, typename = std::enable_if_t<std::is_enum_v<Enum>>>
constexpr std::underlying_type_t<Enum> to_int(const Enum e)
{
	return static_cast<std::underlying_type_t<Enum>>(e);
}

namespace util
{
	template <typename T>
	constexpr T min(const T a, const T b)
	{
		return (a < b) ? a : b;
	}

	template <typename T>
	constexpr T max(const T a, const T b)
	{
		return (a > b) ? a : b;
	}

	template <typename T>
	constexpr T clamp(const T x, const T lower, const T upper)
	{
		return util::min(util::max(x, lower), upper);
	}

	template <typename T>
	constexpr T sgn(const T a)
	{
		return (T(0) < a) - (a < T(0));
	}

	template <typename T>
	constexpr std::make_unsigned_t<T> abs(const T a)
	{
		return a >= T(0) ? a : -a;
	}
}

//
// Adapted from:
// https://stackoverflow.com/questions/35008089/
//

template <typename T, std::size_t S, std::size_t ...Next>
struct array
{
	using next_type = typename array<T, Next...>::type;
	using type = std::array<next_type, S>;
};

template <typename T, std::size_t S>
struct array<T, S>
{
	using type = std::array<T, S>;
};

template <typename T, std::size_t I, std::size_t ...Next>
using array_t = typename array<T, I, Next...>::type;

//
// Basic definitions
//

constexpr auto Colours = 2;
enum Colour : bool
{
	White, Black
};

constexpr auto PieceTypes = 6;
enum PieceType : std::uint8_t
{
	Pawn, Knight, Bishop, Rook, Queen, King
};

constexpr Colour operator~(const Colour colour)
{
	return static_cast<Colour>(colour ^ Black);
}

constexpr std::string_view PieceTypeChars = "pnbrqk";
constexpr std::string_view PieceChars = "PpNnBbRrQqKk";

constexpr auto Files = 8;
enum class File : std::uint8_t
{
	A, B, C, D, E, F, G, H
};

constexpr auto Ranks = 8;
enum class Rank : std::uint8_t
{
	One, Two, Three, Four, Five, Six, Seven, Eight
};

constexpr auto Squares = 64;
enum class Square : std::uint8_t
{
	A1, B1, C1, D1, E1, F1, G1, H1,
	A2, B2, C2, D2, E2, F2, G2, H2,
	A3, B3, C3, D3, E3, F3, G3, H3,
	A4, B4, C4, D4, E4, F4, G4, H4,
	A5, B5, C5, D5, E5, F5, G5, H5,
	A6, B6, C6, D6, E6, F6, G6, H6,
	A7, B7, C7, D7, E7, F7, G7, H7,
	A8, B8, C8, D8, E8, F8, G8, H8,
	Invalid
};

using Direction = std::int8_t;

constexpr Direction North		= 8;
constexpr Direction South		= -North;
constexpr Direction East		= 1;
constexpr Direction West		= -East;
constexpr Direction NorthEast	= North + East;
constexpr Direction NorthWest	= North + West;
constexpr Direction SouthEast	= South + East;
constexpr Direction SouthWest	= South + West;

constexpr bool is_valid(const File file)
{
	return file <= File::H;
}

constexpr bool is_valid(const Rank rank)
{
	return rank <= Rank::Eight;
}

constexpr bool is_valid(const Square sq)
{
	return sq <= Square::H8;
}

constexpr File file_of(const Square sq)
{
	return static_cast<File>(to_int(sq) % Files);
}

constexpr Rank rank_of(const Square sq)
{
	return static_cast<Rank>(to_int(sq) / Ranks);
}

constexpr Square make_square(const File file, const Rank rank)
{
	return static_cast<Square>(to_int(file) + to_int(rank) * Ranks);
}

constexpr File parse_file(const char c)
{
	return static_cast<File>(c - 'a');
}

constexpr Rank parse_rank(const char c)
{
	return static_cast<Rank>(c - '1');
}

constexpr Square parse_square(std::string_view s)
{
	return make_square(parse_file(s[0]), parse_rank(s[1]));
}

constexpr char to_char(const File file)
{
	return char('a' + to_int(file));
}

constexpr char to_char(const Rank rank)
{
	return char('1' + to_int(rank));
}

template <> struct fmt::formatter<File>
{
	template <typename ParseContext> constexpr auto parse(ParseContext &ctx)
	{
		return ctx.begin();
	}

	template <typename FormatContext>
	constexpr auto format(const File file, FormatContext &ctx)
	{
		return format_to(ctx.out(), "{}", to_char(file));
	}
};

template <> struct fmt::formatter<Rank>
{
	template <typename ParseContext> constexpr auto parse(ParseContext &ctx)
	{
		return ctx.begin();
	}

	template <typename FormatContext>
	constexpr auto format(const Rank rank, FormatContext &ctx)
	{
		return format_to(ctx.out(), "{}", to_char(rank));
	}
};

template <> struct fmt::formatter<Square>
{
	template <typename ParseContext> constexpr auto parse(ParseContext &ctx)
	{
		return ctx.begin();
	}

	template <typename FormatContext>
	constexpr auto format(const Square sq, FormatContext &ctx)
	{
		return is_valid(sq)	? format_to(ctx.out(), "{}{}", file_of(sq), rank_of(sq))
							: format_to(ctx.out(), "-");
	}
};

constexpr std::uint8_t rank_distance(const Square a, const Square b)
{
	return util::abs(int(rank_of(a)) - int(rank_of(b)));
}

constexpr std::uint8_t file_distance(const Square a, const Square b)
{
	return util::abs(int(file_of(a)) - int(file_of(b)));
}

constexpr std::uint8_t distance(const Square a, const Square b)
{
	return util::max(rank_distance(a, b), file_distance(a, b));
}

#define ENABLE_ARITHMETIC_OPERATORS(A)                                                             \
constexpr A operator+(const A a, const int b) { return static_cast<A>(static_cast<int>(a) + b); }  \
constexpr A operator-(const A a, const int b) { return static_cast<A>(static_cast<int>(a) - b); }  \
constexpr A operator*(const A a, const int b) { return static_cast<A>(static_cast<int>(a) * b); }  \
constexpr A operator/(const A a, const int b) { return static_cast<A>(static_cast<int>(a) / b); }  \
constexpr A operator%(const A a, const int b) { return static_cast<A>(static_cast<int>(a) % b); }  \
constexpr A &operator+=(A &a, const int b) { return a = a + b; }                                   \
constexpr A &operator-=(A &a, const int b) { return a = a - b; }                                   \
constexpr A &operator++(A &a) { return a += 1; }                                                   \
constexpr A &operator--(A &a) { return a -= 1; }

ENABLE_ARITHMETIC_OPERATORS(File)
ENABLE_ARITHMETIC_OPERATORS(Rank)
ENABLE_ARITHMETIC_OPERATORS(Square)

#undef ENABLE_ARITHMETIC_OPERATORS

union CastlingRights
{
	std::uint8_t all : 4;

	struct { std::uint8_t white : 2, black : 2; };
	struct
	{
		std::uint8_t white_oo : 1, white_ooo : 1;
		std::uint8_t black_oo : 1, black_ooo : 1;
	};
};

constexpr CastlingRights NoCastling    = {0b0000};
constexpr CastlingRights AllCastling   = {0b1111};
constexpr CastlingRights WhiteCastling = {0b0011};
constexpr CastlingRights BlackCastling = {0b1100};
constexpr CastlingRights ShortCastling = {0b0101};
constexpr CastlingRights LongCastling  = {0b1010};
constexpr CastlingRights WhiteShortCastling {WhiteCastling.all & ShortCastling.all};
constexpr CastlingRights BlackShortCastling {BlackCastling.all & ShortCastling.all};
constexpr CastlingRights WhiteLongCastling  {WhiteCastling.all & LongCastling.all};
constexpr CastlingRights BlackLongCastling  {BlackCastling.all & LongCastling.all};

constexpr CastlingRights castling_rights(const Colour us, const bool oo)
{
	if (us == White && oo)
		return WhiteShortCastling;
	else if (us == White)
		return WhiteLongCastling;
	else if (oo)
		return BlackShortCastling;
	else
		return BlackLongCastling;
}

constexpr Square castling_king_dest(const Colour us, const bool oo)
{
	return make_square(oo ? File::G : File::C, us == White ? Rank::One : Rank::Eight);
}

constexpr Square castling_rook_source(const Colour us, const bool oo)
{
	return make_square(oo ? File::H : File::A, us == White ? Rank::One : Rank::Eight);
}

constexpr Square castling_rook_dest(const Colour us, const bool oo)
{
	return make_square(oo ? File::F : File::D, us == White ? Rank::One : Rank::Eight);
}

static constexpr std::array CastlingRightsBySquare
{
	WhiteLongCastling, NoCastling, NoCastling, NoCastling, WhiteCastling, NoCastling, NoCastling, WhiteShortCastling,
	NoCastling, NoCastling, NoCastling, NoCastling, NoCastling, NoCastling, NoCastling, NoCastling,
	NoCastling, NoCastling, NoCastling, NoCastling, NoCastling, NoCastling, NoCastling, NoCastling,
	NoCastling, NoCastling, NoCastling, NoCastling, NoCastling, NoCastling, NoCastling, NoCastling,
	NoCastling, NoCastling, NoCastling, NoCastling, NoCastling, NoCastling, NoCastling, NoCastling,
	NoCastling, NoCastling, NoCastling, NoCastling, NoCastling, NoCastling, NoCastling, NoCastling,
	NoCastling, NoCastling, NoCastling, NoCastling, NoCastling, NoCastling, NoCastling, NoCastling,
	BlackLongCastling, NoCastling, NoCastling, NoCastling, BlackCastling, NoCastling, NoCastling, BlackShortCastling
};

constexpr CastlingRights castling_rights(const Square sq)
{
	return CastlingRightsBySquare[to_int(sq)];
}

//
// Bitboards, part 1
//  Basic definitions
//

using Bitboard = std::uint64_t;

constexpr auto AllBB = ~Bitboard(0);
constexpr auto OneBB = Bitboard(1);

constexpr auto Rank1BB = ~(AllBB << Ranks);
constexpr auto Rank8BB = Rank1BB << (Ranks * 7u);

constexpr auto FileABB = Bitboard(0x101010101010101);
constexpr auto FileHBB = FileABB << 7u;

#if defined(USE_SQUARE_BB)

constexpr array_t<Bitboard, Squares> make_square_lut()
{
	array_t<Bitboard, Squares> lut {};

	for (auto sq = Square::A1; sq <= Square::H8; ++sq)
		lut[to_int(sq)] = 1ull << to_int(sq);

	return lut;
}

static constexpr auto SquareBB = make_square_lut();
constexpr Bitboard square_bb(const Square sq)
{
	ASSERT(is_valid(sq));
	return SquareBB[to_int(sq)];
}

#else
constexpr Bitboard square_bb(const Square sq)
{
	ASSERT(is_valid(sq));
	return 1ull << to_int(sq);
}
#endif

// Convert a file to its corresponding bitboard
constexpr Bitboard file_bb(const File f)
{
	return FileABB << to_int(f);
}

constexpr Bitboard file_bb(const Square sq)
{
	return file_bb(file_of(sq));
}

// Convert a rank to its corresponding bitboard
constexpr Bitboard rank_bb(const Rank r)
{
	return Rank1BB << (Files * to_int(r));
}

constexpr Bitboard rank_bb(const Square sq)
{
	return rank_bb(rank_of(sq));
}

#define ENABLE_BITBOARD_OPERATORS(Type, Convert)                                                   \
constexpr Bitboard operator&(const Bitboard bb, const Type x) { return bb & Convert(x); }          \
constexpr Bitboard operator|(const Bitboard bb, const Type x) { return bb | Convert(x); }          \
constexpr Bitboard operator^(const Bitboard bb, const Type x) { return bb ^ Convert(x); }          \
                                                                                                   \
constexpr Bitboard &operator&=(Bitboard &bb, const Type x) { return bb &= Convert(x); }            \
constexpr Bitboard &operator|=(Bitboard &bb, const Type x) { return bb |= Convert(x); }            \
constexpr Bitboard &operator^=(Bitboard &bb, const Type x) { return bb ^= Convert(x); }            \
                                                                                                   \
constexpr Bitboard operator&(const Type x, const Bitboard bb) { return bb & x; }                   \
constexpr Bitboard operator|(const Type x, const Bitboard bb) { return bb | x; }                   \
constexpr Bitboard operator^(const Type x, const Bitboard bb) { return bb ^ x; }                   \

ENABLE_BITBOARD_OPERATORS(Square, square_bb)
ENABLE_BITBOARD_OPERATORS(File, file_bb)
ENABLE_BITBOARD_OPERATORS(Rank, rank_bb)

// Convert two squares to an equivalent bitboard
constexpr Bitboard square_bb(const Square a, const Square b)
{
	return square_bb(a) | b;
}

// Check if a bitboard has more than one bit set
constexpr bool more_than_one(const Bitboard bb)
{
	// Pop LSB, if bb is still nonzero then
	// there must be more than one set bit.
	return bb & (bb - 1);
}

// Check if a bitboard has exactly one bit set
constexpr bool only_one(const Bitboard bb)
{
	return bb && !more_than_one(bb);
}

//
// Bitboards, part 2
//  Kogge-Stone implementation for sliding piece attacks
// 	 + shifting bitboards w/o wrapping + assorted functions.
//
//  Kogge-Stone functions are used in initialisation of
//   fancy magic/PEXT/PEXT+PDEP bitboards, and are also used
//   to determine attacks from multiple sliding pieces at once.
//

// Shift bitboard in any direction, preventing wrapping.
template <Direction D> constexpr Bitboard shift(const Bitboard &bb);

template <> constexpr Bitboard shift<North>(const Bitboard &bb)
{
	return bb << 8u;
}

template <> constexpr Bitboard shift<South>(const Bitboard &bb)
{
	return bb >> 8u;
}

template <> constexpr Bitboard shift<East>(const Bitboard &bb)
{
	return (bb & ~FileHBB) << 1u;
}

template <> constexpr Bitboard shift<West>(const Bitboard &bb)
{
	return (bb & ~FileABB) >> 1u;
}

template <> constexpr Bitboard shift<NorthEast>(const Bitboard &bb)
{
	return (bb & ~FileHBB) << 9u;
}

template <> constexpr Bitboard shift<SouthWest>(const Bitboard &bb)
{
	return (bb & ~FileABB) >> 9u;
}

template <> constexpr Bitboard shift<NorthWest>(const Bitboard &bb)
{
	return (bb & ~FileABB) << 7u;
}

template <> constexpr Bitboard shift<SouthEast>(const Bitboard &bb)
{
	return (bb & ~FileHBB) >> 7u;
}

template <class = void> constexpr Bitboard walk(const Bitboard &bb)
{
	return bb;
}

// Shifts a bitboard in several directions continuously.
// Example: walk<North, North, East>(A1) = B3
template <Direction Step, Direction... Next> constexpr Bitboard walk(const Bitboard &bb)
{
	return walk<Next...>(shift<Step>(bb));
}

template <class = void> constexpr Bitboard shift_ex(const Bitboard &)
{
	return 0;
}

// Shifts a bitboard in several directions at once
// Example: shift_ex<North, South, East>(E4) = E5 | E3 | F4
template <Direction Step, Direction... Next> constexpr Bitboard shift_ex(const Bitboard &bb)
{
	return shift<Step>(bb) | shift_ex<Next...>(bb);
}

//
// Implementations of Kogge-Stone flood fill + occluded fill.
// https://www.chessprogramming.org/Kogge-Stone_Algorithm
//

// Flood fill in a single direction.
template <Direction D> constexpr Bitboard fill(Bitboard gen);

// Occluded fill in a single direction.
template <Direction D> constexpr Bitboard fill(Bitboard gen, Bitboard pro);

template <> constexpr Bitboard fill<North>(Bitboard gen)
{
	gen |= (gen << 8u);
	gen |= (gen << 16u);
	gen |= (gen << 32u);

	return gen;
}

template <> constexpr Bitboard fill<South>(Bitboard gen)
{
	gen |= (gen >> 8u);
	gen |= (gen >> 16u);
	gen |= (gen >> 32u);

	return gen;
}

template <> constexpr Bitboard fill<East>(Bitboard gen)
{
	constexpr auto a = ~FileABB, b = a & (a << 1u), c = b & (b << 2u);

	gen |= a & (gen << 1u);
	gen |= b & (gen << 2u);
	gen |= c & (gen << 4u);

	return gen;
}

template <> constexpr Bitboard fill<West>(Bitboard gen)
{
	constexpr auto a = ~FileHBB, b = a & (a >> 1u), c = b & (b >> 2u);

	gen |= a & (gen >> 1u);
	gen |= b & (gen >> 2u);
	gen |= c & (gen >> 4u);

	return gen;
}

template <> constexpr Bitboard fill<NorthEast>(Bitboard gen)
{
	constexpr auto a = ~FileABB, b = a & (a << 9u), c = b & (b << 18u);

	gen |= a & (gen << 9u);
	gen |= b & (gen << 18u);
	gen |= c & (gen << 36u);

	return gen;
}

template <> constexpr Bitboard fill<SouthWest>(Bitboard gen)
{
	constexpr auto a = ~FileHBB, b = a & (a >> 9u), c = b & (b >> 18u);

	gen |= a & (gen >> 9u);
	gen |= b & (gen >> 18u);
	gen |= c & (gen >> 36u);

	return gen;
}

template <> constexpr Bitboard fill<NorthWest>(Bitboard gen)
{
	constexpr auto a = ~FileHBB, b = a & (a << 7u), c = b & (b << 14u);

	gen |= a & (gen << 7u);
	gen |= b & (gen << 14u);
	gen |= c & (gen << 28u);

	return gen;
}

template <> constexpr Bitboard fill<SouthEast>(Bitboard gen)
{
	constexpr auto a = ~FileABB, b = a & (a >> 7u), c = b & (b >> 14u);

	gen |= a & (gen >> 7u);
	gen |= b & (gen >> 14u);
	gen |= c & (gen >> 28u);

	return gen;
}

template <> constexpr Bitboard fill<North>(Bitboard gen, Bitboard pro)
{
	gen |= (gen << 8u) & pro;
	pro &= (pro << 8u);
	gen |= (gen << 16u) & pro;
	pro &= (pro << 16u);
	gen |= (gen << 32u) & pro;

	return gen;
}

template <> constexpr Bitboard fill<South>(Bitboard gen, Bitboard pro)
{
	gen |= (gen >> 8u) & pro;
	pro &= (pro >> 8u);
	gen |= (gen >> 16u) & pro;
	pro &= (pro >> 16u);
	gen |= (gen >> 32u) & pro;

	return gen;
}

template <> constexpr Bitboard fill<East>(Bitboard gen, Bitboard pro)
{
	pro &= ~FileABB;

	gen |= (gen << 1u) & pro;
	pro &= (pro << 1u);
	gen |= (gen << 2u) & pro;
	pro &= (pro << 2u);
	gen |= (gen << 4u) & pro;

	return gen;
}

template <> constexpr Bitboard fill<West>(Bitboard gen, Bitboard pro)
{
	pro &= ~FileHBB;

	gen |= (gen >> 1u) & pro;
	pro &= (pro >> 1u);
	gen |= (gen >> 2u) & pro;
	pro &= (pro >> 2u);
	gen |= (gen >> 4u) & pro;

	return gen;
}

template <> constexpr Bitboard fill<NorthEast>(Bitboard gen, Bitboard pro)
{
	pro &= ~FileABB;

	gen |= (gen << 9u) & pro;
	pro &= (pro << 9u);
	gen |= (gen << 18u) & pro;
	pro &= (pro << 18u);
	gen |= (gen << 36u) & pro;

	return gen;
}

template <> constexpr Bitboard fill<SouthWest>(Bitboard gen, Bitboard pro)
{
	pro &= ~FileHBB;

	gen |= (gen >> 9u) & pro;
	pro &= (pro >> 9u);
	gen |= (gen >> 18u) & pro;
	pro &= (pro >> 18u);
	gen |= (gen >> 36u) & pro;

	return gen;
}

template <> constexpr Bitboard fill<NorthWest>(Bitboard gen, Bitboard pro)
{
	pro &= ~FileHBB;

	gen |= (gen << 7u) & pro;
	pro &= (pro << 7u);
	gen |= (gen << 14u) & pro;
	pro &= (pro << 14u);
	gen |= (gen << 28u) & pro;

	return gen;
}

template <> constexpr Bitboard fill<SouthEast>(Bitboard gen, Bitboard pro)
{
	pro &= ~FileABB;

	gen |= (gen >> 7u) & pro;
	pro &= (pro >> 7u);
	gen |= (gen >> 14u) & pro;
	pro &= (pro >> 14u);
	gen |= (gen >> 28u) & pro;

	return gen;
}

template <class = void> constexpr Bitboard ray_attacks(const Bitboard)
{
	return 0;
}

// Determines direction-wise sliding piece attacks
template <Direction D, Direction... Next>
constexpr Bitboard ray_attacks(const Bitboard pieces)
{
	return shift<D>(fill<D>(pieces)) | ray_attacks<Next...>(pieces);
}

template <class = void> constexpr Bitboard ray_attacks(const Bitboard, const Bitboard)
{
	return 0;
}

// Determines direction-wise sliding piece attacks, with occupancy
template <Direction D, Direction... Next>
constexpr Bitboard ray_attacks(const Bitboard pieces, const Bitboard occ)
{
	return shift<D>(fill<D>(pieces, ~occ)) | ray_attacks<Next...>(pieces, occ);
}

// Determines attacks from several squares at once
template <PieceType> constexpr Bitboard attacks_from(Bitboard);

template <> constexpr Bitboard attacks_from<Knight>(Bitboard pieces)
{
	const auto l1 = (pieces >> 1) & 0x7f7f7f7f7f7f7f7f;
	const auto l2 = (pieces >> 2) & 0x3f3f3f3f3f3f3f3f;
	const auto r1 = (pieces << 1) & 0xfefefefefefefefe;
	const auto r2 = (pieces << 2) & 0xfcfcfcfcfcfcfcfc;
	const auto h1 = l1 | r1;
	const auto h2 = l2 | r2;

	return (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8);
}

template <> constexpr Bitboard attacks_from<King>(Bitboard pieces)
{
	auto attacks = shift_ex<West, East>(pieces);
	pieces |= attacks;
	return attacks | shift_ex<North, South>(pieces);
}

template <> constexpr Bitboard attacks_from<Bishop>(Bitboard pieces)
{
	return ray_attacks<NorthEast, SouthEast, SouthWest, NorthWest>(pieces);
}

template <> constexpr Bitboard attacks_from<Rook>(Bitboard pieces)
{
	return ray_attacks<North, East, South, West>(pieces);
}

template <> constexpr Bitboard attacks_from<Queen>(Bitboard pieces)
{
	return attacks_from<PieceType::Bishop>(pieces) | attacks_from<PieceType::Rook>(pieces);
}

// Determines attacks from several squares at once, with occupancy
template <PieceType T> constexpr Bitboard attacks_from(Bitboard pieces, Bitboard)
{
	return attacks_from<T>(pieces);
}

template <> constexpr Bitboard attacks_from<Bishop>(Bitboard pieces, Bitboard occ)
{
	return ray_attacks<NorthEast, SouthEast, SouthWest, NorthWest>(pieces, occ);
}

template <> constexpr Bitboard attacks_from<PieceType::Rook>(Bitboard pieces, Bitboard occ)
{
	return ray_attacks<North, East, South, West>(pieces, occ);
}

template <> constexpr Bitboard attacks_from<PieceType::Queen>(Bitboard pieces, Bitboard occ)
{
	return attacks_from<PieceType::Bishop>(pieces, occ)
		 | attacks_from<PieceType::Rook  >(pieces, occ);
}

//
// Bitboards: part 3
// Line between / line connecting
//

constexpr array_t<Bitboard, Squares, Squares> make_line_connecting_lut()
{
	array_t<Bitboard, Squares, Squares> lut {};

	for (auto sq = Square::A1; sq <= Square::H8; ++sq)
	{
		for (auto sq2 = Square::A1; sq2 <= Square::H8; ++sq2)
		{
			const auto i = to_int(sq), j = to_int(sq2);
			const auto a = square_bb(sq), b = square_bb(sq2);

			if (attacks_from<Bishop>(a) & b)
				lut[i][j] = (attacks_from<Bishop>(a) & attacks_from<Bishop>(b)) | a | b;
			else if (attacks_from<Rook>(a) & b)
				lut[i][j] = (attacks_from<Rook>(a) & attacks_from<Rook>(b)) | a | b;
		}
	}

	return lut;
}

static constexpr auto LineBB = make_line_connecting_lut();
constexpr Bitboard line_connecting(Square sq, Square sq2)
{
	ASSERT(is_valid(sq));
	ASSERT(is_valid(sq2));
	return LineBB[to_int(sq)][to_int(sq2)];
}

#if defined(USE_BETWEEN_BB)
constexpr array_t<Bitboard, Squares, Squares> make_line_between_lut()
{
	array_t<Bitboard, Squares, Squares> lut {};

	for (auto sq = Square::A1; sq <= Square::H8; ++sq)
	{
		for (auto sq2 = Square::A1; sq2 <= Square::H8; ++sq2)
		{
			const auto i = to_int(sq), j = to_int(sq2);

			lut[i][j] =	line_connecting(sq, sq2)
					  & ((AllBB << i) ^ (AllBB << j));
			lut[i][j] &= (lut[i][j] - 1u);
		}
	}

	return lut;
}

static constexpr auto BetweenBB = make_line_between_lut();
constexpr Bitboard line_between(Square sq, Square sq2)
{
	ASSERT(is_valid(sq));
	ASSERT(is_valid(sq2));
	return BetweenBB[to_int(sq)][to_int(sq2)];
}
#else
constexpr Bitboard line_between(Square sq, Square sq2)
{
	ASSERT(is_valid(sq));
	ASSERT(is_valid(sq2));

	const auto bb = line_connecting(sq, sq2) & ((AllBB << to_int(sq)) ^ (AllBB << to_int(sq2)));
	return bb & (bb - 1);
}
#endif

constexpr bool aligned(const Square a, const Square b)
{
	return line_connecting(a, b);
}

constexpr bool aligned(const Square a, const Square b, const Square c)
{
	return line_connecting(a, b) & c;
}

//
// Bitboards, part 4
//  Attack generation using Kogge-Stone/Fancy magic/PEXT/PEXT+PDEP
//

// Pawn attacks from a bitboard
template <Colour Us>
constexpr Bitboard pawn_attacks(const Bitboard pawns)
{
	return Us == Colour::White ? shift_ex<NorthWest, NorthEast>(pawns)
							   : shift_ex<SouthWest, SouthEast>(pawns);
}

// Pawn attacks from a bitboard
constexpr Bitboard pawn_attacks(const Colour us, const Bitboard pawns)
{
	return us == Colour::White ? pawn_attacks<Colour::White>(pawns)
							   : pawn_attacks<Colour::Black>(pawns);
}

// Pawn attacks from a single square
constexpr Bitboard pawn_attacks(const Colour us, const Square sq)
{
	return pawn_attacks(us, square_bb(sq));
}

template <PieceType T>
constexpr array_t<Bitboard, Squares> make_attacks_lut()
{
	array_t<Bitboard, Squares> lut {};

	for (auto sq = Square::A1; sq <= Square::H8; ++sq)
		lut[to_int(sq)] = attacks_from<T>(square_bb(sq));

	return lut;
}

// Determines attacks from a given square
template <PieceType T>
static constexpr auto AttacksBB = make_attacks_lut<T>();

template <PieceType>
constexpr Bitboard attacks_from(const Square);

template <>
constexpr Bitboard attacks_from<Knight>(const Square sq)
{
#if defined(USE_KNIGHT_BB)
	return AttacksBB<Knight>[to_int(sq)];
#else
	return attacks_from<Knight>(square_bb(sq));
#endif
}

template <>
constexpr Bitboard attacks_from<King>(const Square sq)
{
#if defined(USE_KING_BB)
	return AttacksBB<King>[to_int(sq)];
#else
	return attacks_from<King>(square_bb(sq));
#endif
}

template <>
constexpr Bitboard attacks_from<Bishop>(const Square sq)
{
#if defined(USE_BISHOP_BB)
	return AttacksBB<Bishop>[to_int(sq)];
#else
	return attacks_from<Bishop>(square_bb(sq));
#endif
}

template <>
constexpr Bitboard attacks_from<Rook>(const Square sq)
{
#if defined(USE_ROOK_BB)
	return AttacksBB<Rook>[to_int(sq)];
#else
	return attacks_from<Rook>(square_bb(sq));
#endif
}

template <> 
constexpr Bitboard attacks_from<Queen>(const Square sq)
{
#if defined(USE_QUEEN_BB)
	return AttacksBB<Queen>[to_int(sq)];
#else
	return attacks_from<Bishop>(sq) | attacks_from<Rook>(sq);
#endif
}

// Determines attacks from a given square, with occupancy
template <PieceType T>
constexpr Bitboard attacks_from(const Square sq, const Bitboard)
{
	return attacks_from<T>(sq);
}

#if defined(USE_KOGGE)
template <>
constexpr Bitboard attacks_from<Bishop>(const Square sq, const Bitboard occ)
{
	ASSERT(is_valid(sq));
	return attacks_from<Bishop>(square_bb(sq), occ);
}

template <>
constexpr Bitboard attacks_from<Rook>(const Square sq, const Bitboard occ)
{
	ASSERT(is_valid(sq));
	return attacks_from<Rook>(square_bb(sq), occ);
}

template <>
constexpr Bitboard attacks_from<Queen>(const Square sq, const Bitboard occ)
{
	return attacks_from<Bishop>(sq, occ) | attacks_from<Rook>(sq, occ);
}
#else
template <PieceType>
constexpr Bitboard sliding_attacks(const Square, const Bitboard);

template <>
constexpr Bitboard sliding_attacks<Bishop>(const Square sq, const Bitboard occ)
{
	return ray_attacks<NorthEast, SouthEast, SouthWest, NorthWest>(square_bb(sq), occ);
}

template <>
constexpr Bitboard sliding_attacks<Rook>(const Square sq, const Bitboard occ)
{
	return ray_attacks<North, East, South, West>(square_bb(sq), occ);
}

#	if defined(USE_FANCY)
constexpr array_t<Bitboard, Squares> PrecomputedBishopMagics
{
	0x04408a8084008180, 0x5c20220a02023410, 0x9004010202080000, 0x0020a90100440021,
	0x2002021000400412, 0x900a022220004014, 0x006084886030a000, 0x6900602216104000,
	0x1500100238890400, 0x0430a08182060040, 0x0100104102002010, 0x2084040404814004,
	0x0080040420040020, 0x8000010420051000, 0x60404e0111084000, 0x0102010118010400,
	0x2010910a10010810, 0x0009000401080a01, 0x020a041002220060, 0x0006880802004100,
	0x080c100202021800, 0x1802012108021a00, 0x000281620a101310, 0x2810400080480800,
	0x8c1008014108410c, 0x0014056020210400, 0x0a40880c10004214, 0x4084080100202040,
	0x80250040cc044000, 0x1008490222008200, 0x060a0400008c0110, 0x2a01002001028861,
	0x400220a001100301, 0x910110480002d800, 0x0002081200040c10, 0x0090220080080480,
	0x0210020010020104, 0x4404410e00004800, 0x4001640400210100, 0x0024410440320050,
	0x0009012011102000, 0x5006012420000204, 0x0031101804002800, 0x1000806081200800,
	0x815106200a022500, 0xa010604080200500, 0x0605100411000041, 0x0148020a81340208,
	0x800d1402a0042010, 0x0802011088940344, 0x0180011088040000, 0x0814380084040088,
	0x00004a0963040097, 0x0464220e6a420081, 0x0020610102188000, 0x0204c10801010008,
	0x0620240108080220, 0x0004008a08210404, 0x200c001302949000, 0x2a00044080208804,
	0x0040048a91820210, 0x00100c2005012200, 0x1420900310040884, 0x805220940900c100
};

constexpr array_t<Bitboard, Squares> PrecomputedRookMagics
{
	0x0080001084284000, 0x2140022000100040, 0x0d00082001004010, 0x0900100100882084,
	0x0080080080040002, 0xc1800600010c0080, 0x0600040100a80200, 0x4100020088482900,
	0x1089800040008020, 0x4000404010002000, 0x4030808020001000, 0x0000801000080080,
	0x0006002004483200, 0x0021000803000400, 0x0221000200048100, 0x0001001190470002,
	0x0008888000400030, 0x10c0002020081000, 0x0250420018208200, 0x00800a0010422200,
	0x9001050011014800, 0x0020808002000400, 0x0808440010c80102, 0x0586020020410084,
	0x2080005040002000, 0x2020008101004000, 0x0420008080100020, 0x0008001010020100,
	0x2400080080800400, 0x0b62040080800200, 0x0086000200040108, 0x080408a200005104,
	0x1000400028800080, 0x2c01c02009401000, 0x0048882000801000, 0x400090000b002100,
	0x2214100801000500, 0x0000800400800200, 0x1828420124003088, 0x000035144a000084,
	0x0000400080208000, 0x8010044320044004, 0x0100200011010040, 0x0802002008420010,
	0x418c000800110100, 0x0002002010040400, 0x0002502841040022, 0x010040c08402001f,
	0x02204000800c2880, 0x0c10002000400040, 0x4500201200804200, 0x0e50010822910100,
	0x1200040008008080, 0x080e000400028080, 0x0000018208100400, 0x0440004400a10200,
	0xd308204011020082, 0xc447102040090181, 0x00060020420a8012, 0x00402e40180e00e2,
	0x1000054800110095, 0x0241000400020801, 0x0001300100880244, 0x0061122401004a86
};
#	endif

struct MagicInfo
{
#	if defined(USE_FANCY)
		Bitboard mask, magic;
		std::uint8_t shift;
#	elif defined(USE_PEXT)
		Bitboard mask;
#	elif defined(USE_PDEP)
		Bitboard mask, postmask;
#	endif

	std::size_t offset;
};

// Stores magic info for each square + attack database
template <PieceType T>
struct MagicTable
{
	static constexpr auto Size = T == Rook ? 102400 : 5248;

	array_t<MagicInfo, Squares> magic_info;

#	if defined(USE_PDEP)
	array_t<std::uint16_t, Size> attack_table;
#	else
	array_t<Bitboard, Size> attack_table;
#	endif

	MagicTable() : magic_info(), attack_table()
	{
		std::size_t size = 0;
		Bitboard edges = 0, attacks = 0, occ = 0;

		for (auto sq = Square::A1; sq <= Square::H8; ++sq)
		{
			edges = ((Rank1BB | Rank8BB) & ~rank_bb(sq))
				  | ((FileABB | FileHBB) & ~file_bb(sq));

			attacks = sliding_attacks<T>(sq, 0);

#	if defined(USE_PDEP)
			magic_info[to_int(sq)].postmask = attacks;
#	endif
			magic_info[to_int(sq)].mask = attacks & ~edges;

#	if defined(USE_FANCY)
			magic_info[to_int(sq)].shift = 64u - popcount_generic(magic_info[to_int(sq)].mask);
			magic_info[to_int(sq)].magic = T == Rook ? PrecomputedRookMagics[to_int(sq)]
											 : PrecomputedBishopMagics[to_int(sq)];
#	endif

			magic_info[to_int(sq)].offset = sq == Square::A1 ? 0
											 : magic_info[to_int(sq - 1)].offset + size;

			occ  = 0;
			size = 0;
			do
			{
				attacks = sliding_attacks<T>(sq, occ);

#	if defined(USE_PDEP)
				attack_table[magic_info[to_int(sq)].offset + index(sq, occ)] = pext(attacks, magic_info[to_int(sq)].postmask);
#	else
				attack_table[magic_info[to_int(sq)].offset + index(sq, occ)] = attacks;
#	endif

				++size;
				occ = (occ - magic_info[to_int(sq)].mask) & magic_info[to_int(sq)].mask;
			}
			while (occ);
		}
	}

	std::size_t index(const Square sq, const Bitboard occ) const
	{
#	if defined(USE_FANCY)
		return ((occ & magic_info[to_int(sq)].mask) * magic_info[to_int(sq)].magic) >> magic_info[to_int(sq)].shift;
#	elif defined(USE_PEXT) || defined(USE_PDEP)
		return pext(occ, magic_info[to_int(sq)].mask);
#	endif
	}

	Bitboard probe(const Square sq, const Bitboard occ) const
	{
		const auto attacks = attack_table[magic_info[to_int(sq)].offset + index(sq, occ)];

#	if defined(USE_PDEP)
		return pdep(attacks, magic_info[to_int(sq)].postmask);
#	else
		return attacks;
#	endif
	}
};

static MagicTable<Bishop> bishop_magic_table {};
static MagicTable<Rook> rook_magic_table {};

template <>
inline Bitboard attacks_from<Bishop>(const Square sq, const Bitboard occ)
{
	return bishop_magic_table.probe(sq, occ);
}

template <>
inline Bitboard attacks_from<Rook>(const Square sq, const Bitboard occ)
{
	return rook_magic_table.probe(sq, occ);
}

template <>
inline Bitboard attacks_from<Queen>(const Square sq, const Bitboard occ)
{
	return attacks_from<Bishop>(sq, occ) | attacks_from<Rook>(sq, occ);
}

#endif

//
// Bitboards, part 5
//  Misc. functions
//

inline std::string to_string(const Bitboard bb)
{
	std::string s = "/---------------\\\n";

	for (auto rank = Rank::Eight; is_valid(rank); --rank)
	{
		for (auto file = File::A; is_valid(file); ++file)
		{
			s += (bb & make_square(file, rank)) ? "|1" : "|0";
		}
		s += "|\n";
	}

	s += "\\---------------/\n";

	return s;
}

constexpr Bitboard castling_king_path(const Colour us, const bool oo)
{
	const auto ksq = make_square(File::E, us == White ? Rank::One : Rank::Eight);
	const auto kto = castling_king_dest(us, oo);

	return line_between(ksq, kto) | kto;
}

constexpr Bitboard castling_rook_path(const Colour us, const bool oo)
{
	const auto rsq = castling_rook_source(us, oo);
	const auto rto = castling_rook_dest(us, oo);

	return line_between(rsq, rto) | rto;
}

//
// Board structure
//

struct Board
{
	constexpr Board()
		: white_pieces(0), black_pieces(0), pawns(0), knights(0), bishops_queens(0),
		  rooks_queens(0), white_king(Square::Invalid), black_king(Square::Invalid),
		  castling_rights(NoCastling), side(White), en_passant(Square::Invalid) {};

	Bitboard white_pieces, black_pieces;
	Bitboard pawns, knights, bishops_queens, rooks_queens;
	Square white_king, black_king;

	CastlingRights castling_rights;
	Colour side;
	Square en_passant;
};

inline int parse_fen(Board &board, std::string_view fen)
{
	std::size_t pos;
	char c;

	auto file = File::A;
	auto rank = Rank::Eight;

	// Parse piece placement
	// Iterate through each character of FEN until we reach a space or the end
	for (pos = 0, c = fen[pos]; pos < fen.size(); ++pos, c = fen[pos])
	{
		if (std::isdigit(c))
		{
			// Use '1' instead of '0' since pos will be incremented after this iteration anyway
			const int skip = c - '1';

			// Check for c == '0', and ensure skip is within range
			// (i.e. file won't be off the board or decremented)
			if (skip < 0 || skip > (1 + to_int(File::H) - to_int(file)))
				return 11;

			file += skip;
		}
		else if (c == '/')
		{
			// Reset file, move onto next rank
			file = File::A;
			--rank;

			// Continue to avoid incrementing file
			continue;
		}
		else if (auto p = PieceChars.find(c); p != std::string::npos)
		{
			const auto colour = std::isupper(c) ? White : Black;
			c = std::tolower(c);

			const auto sq = make_square(file, rank);
			
			switch (c)
			{
			case 'p':
				board.pawns |= sq;
				break;
			case 'n':
				board.knights |= sq;
				break;
			case 'b':
				board.bishops_queens |= sq;
				break;
			case 'r':
				board.rooks_queens |= sq;
				break;
			case 'q':
				board.bishops_queens |= sq;
				board.rooks_queens |= sq;
				break;
			case 'k':
				(colour == White ? board.white_king : board.black_king) = sq;
				break;
			default:
				std::abort();
			}

			(colour == White ? board.white_pieces : board.black_pieces) |= sq;
		}
		else if (c == ' ')
			break;
		else
			return 19; // Unexpected/unknown character

		++file;
	}

	// Parse side to move
	pos += 1;
	if (c = std::tolower(fen[pos]); c == 'w' || c == 'b')
		board.side = c == 'w' ? White : Black;
	else
		return 29; // Unexpected/unknown character
	pos += 2;

	// Parse castling rights
	if (c == '-')
		board.castling_rights = NoCastling;
	else
	{
		board.castling_rights = NoCastling;

		for (c = fen[pos]; c != ' ' && pos < fen.size(); ++pos, c = fen[pos])
		{
			CastlingRights rights = NoCastling;

			switch (c)
			{
			case 'K':
				rights = WhiteShortCastling;
				break;
			case 'Q':
				rights = WhiteLongCastling;
				break;
			case 'k':
				rights = BlackShortCastling;
				break;
			case 'q':
				rights = BlackLongCastling;
				break;
			case '-':
				break;
			default:
				return 39; // Unknown character
			}

			board.castling_rights.all |= rights.all;
		}
	}

	// Parse en passant
	pos += 1;
	if (c = fen[pos]; c == '-')
		board.en_passant = Square::Invalid;
	else
	{
		const auto sq = make_square(parse_file(c), parse_rank(fen[++pos]));
		if (!is_valid(sq))
			return 41; // Invalid en passant square
		
		board.en_passant = sq;
	}
	pos += 2;

	return 0;
}

inline std::string to_string(const Board &board)
{
	std::string s = "/---------------\\\n";

	const auto occ = board.white_pieces | board.black_pieces;

	for (auto rank = Rank::Eight; is_valid(rank); --rank)
	{
		for (auto file = File::A; is_valid(file); ++file)
		{
			s += '|';
			const auto sq = make_square(file, rank);
			if ((occ & sq) == 0)
				s += '-';
			else
			{
				char c;

				if (sq & board.pawns)
					c = 'p';
				else if (sq & board.knights)
					c = 'n';
				else if (sq & board.bishops_queens & board.rooks_queens)
					c = 'q';
				else if (sq & board.bishops_queens)
					c = 'b';
				else if (sq & board.rooks_queens)
					c = 'r';
				else
					c = 'k';

				s += sq & board.white_pieces ? char(std::toupper(c)) : c;
			}
		}
		s += "|\n";
	}

	s += "\\---------------/\n";
	s += fmt::format("Side to move: {}\n", board.side == White ? "white" : "black");
	s += fmt::format("En passant  : {}\n", board.en_passant);
	s += fmt::format("Castling    : {:04b}\n", board.castling_rights.all);

	return s;
}

constexpr Board startpos()
{
	Board board;

	board.white_king = Square::E1;
	board.black_king = Square::E8;

	board.white_pieces = 0xffff;
	board.black_pieces = 0xffff000000000000;

	board.pawns = 0xff00000000ff00;
	board.knights = 0x4200000000000042;
	board.bishops_queens = 0x2c0000000000002c;
	board.rooks_queens = 0x8900000000000089;

	board.castling_rights = AllCastling;
	board.side = White;
	board.en_passant = Square::Invalid;

	return board;
}

template <Colour Us>
constexpr Bitboard checks(const Board &board)
{
	const auto ksq			= Us == White ? board.white_king : board.black_king;
	const auto their_pieces	= Us == White ? board.black_pieces : board.white_pieces;

	const auto occ = board.white_pieces | board.black_pieces;

	return ((attacks_from<Bishop>(ksq, occ) & board.bishops_queens)
		|   (attacks_from<Rook>  (ksq, occ) & board.rooks_queens)
		|   (attacks_from<Knight>(ksq) & board.knights)
		|   (pawn_attacks(Us, ksq) & board.pawns)) & their_pieces;
}

template <Colour Us>
constexpr Bitboard unsafe_squares(const Board &board)
{
	constexpr auto Them		= ~Us;
	const auto ksq			= Us == White ? board.white_king : board.black_king;
	const auto eksq			= Us == White ? board.black_king : board.white_king;
	const auto their_pieces	= Us == White ? board.black_pieces : board.white_pieces;

	const auto occ = (board.white_pieces | board.black_pieces) ^ ksq;

	return attacks_from<Bishop>(board.bishops_queens & their_pieces, occ)
		|  attacks_from<Rook>  (board.rooks_queens & their_pieces, occ)
		|  attacks_from<Knight>(board.knights & their_pieces)
		|  attacks_from<King>  (eksq)
		|  pawn_attacks<Them>  (board.pawns & their_pieces);
}

template <Colour us>
inline Bitboard pinned_pieces(const Board &board)
{
	const auto ksq		= us == White ? board.white_king : board.black_king;
	const auto friendly	= us == White ? board.white_pieces : board.black_pieces;
	const auto enemy	= us == White ? board.black_pieces : board.white_pieces;
	const auto occ		= friendly | enemy;

	auto candidates = ((attacks_from<Bishop>(ksq) & board.bishops_queens)
					|  (attacks_from<Rook>  (ksq) & board.rooks_queens)) & enemy;

	Bitboard pinned = 0;

	while (candidates)
	{
		const auto candidate = static_cast<Square>(lsb(candidates));

		// If there is only one piece between ksq and candidate_sq, then
		// that piece is pinned if it is the same colour as the piece on sq.
		const auto maybe_pinned = line_between(ksq, candidate) & occ;
		if (only_one(maybe_pinned))
			pinned |= maybe_pinned & friendly;

		candidates &= (candidates - 1);
	}

	return pinned;
}

template <Colour Us, PieceType T, PieceType Promotion = Pawn>
void do_move(Board &board, const Square from, const Square to)
{
	const auto to_bb = square_bb(to);
	const auto mask = to_bb | from;

	const auto en_passant = board.en_passant;

	// Update state
	board.side = ~Us;
	board.en_passant = Square::Invalid;

	// Clear destination square
	board.pawns &= ~to_bb;
	board.knights &= ~to_bb;
	board.bishops_queens &= ~to_bb;
	board.rooks_queens &= ~to_bb;

	// Move the piece
	if (T == Pawn)
	{
		board.pawns ^= from;

		if (Promotion == Knight)
			board.knights |= to_bb;
		else if (Promotion == Bishop)
			board.bishops_queens |= to_bb;
		else if (Promotion == Rook)
			board.rooks_queens |= to_bb;
		else if (Promotion == Queen)
		{
			board.bishops_queens |= to_bb;
			board.rooks_queens |= to_bb;
		}
		else
		{
			board.pawns ^= to;

			constexpr auto Down = Us == White ? South : North;

			// En passant
			if (to == en_passant)
			{
				const auto ep_mask = shift<Down>(to_bb);
				board.pawns &= ~ep_mask;

				if (Us == White)
					board.black_pieces &= ~ep_mask;
				else if (Us == Black)
					board.white_pieces &= ~ep_mask;
			}
			else if (distance(from, to) == 2)
				board.en_passant = to + Down;
		}
	}
	else if (T == Knight)
		board.knights ^= mask;
	else if (T == Bishop)
	{
		board.bishops_queens ^= mask;

		if (board.rooks_queens & mask)
			board.rooks_queens ^= mask;
	}
	else if (T == Rook)
	{
		board.rooks_queens ^= mask;

		if (board.bishops_queens & mask)
			board.bishops_queens ^= mask;
	}
	else if (T == Queen)
	{
		board.bishops_queens ^= mask;
		board.rooks_queens ^= mask;
	}
	else if (T == King)
	{
		if (distance(from, to) == 2)
		{
			const auto oo = to > from;
			const auto rook_mask = square_bb(castling_rook_source(Us, oo),
											 castling_rook_dest(Us, oo));

			board.rooks_queens ^= rook_mask;

			if (Us == White)
				board.white_pieces ^= rook_mask;
			else if (Us == Black)
				board.black_pieces ^= rook_mask;
		}

		if (Us == White)
			board.white_king = to;
		else if (Us == Black)
			board.black_king = to;
	}

	if (Us == White)
	{
		board.white_pieces ^= mask;
		board.black_pieces &= ~to_bb;
	}
	else if (Us == Black)
	{
		board.black_pieces ^= mask;
		board.white_pieces &= ~to_bb;
	}

	board.castling_rights.all &= ~castling_rights(from).all;
	board.castling_rights.all &= ~castling_rights(to).all;
}

inline int parse_and_push_uci(Board &board, std::string_view uci)
{
	if (uci.size() != 4 && uci.size() != 5)
		return 3;

	const auto from = parse_square(uci.substr(0, 2)), to = parse_square(uci.substr(2, 2));

	if (!is_valid(from))
		return 4;
	
	if (!is_valid(to))
		return 5;

	auto promotion = Pawn;
	if (uci.size() == 5)
		promotion = static_cast<PieceType>(PieceTypeChars.find(uci[4]));

	if (promotion != Pawn)
	{
		if (promotion == Knight)
			board.side == White ? do_move<White, Knight>(board, from, to)
								: do_move<Black, Knight>(board, from, to);
		else if (promotion == Bishop)
			board.side == White ? do_move<White, Bishop>(board, from, to)
								: do_move<Black, Bishop>(board, from, to);
		else if (promotion == Rook)
			board.side == White ? do_move<White, Rook>(board, from, to)
								: do_move<Black, Rook>(board, from, to);
		else if (promotion == Queen)
			board.side == White ? do_move<White, Queen>(board, from, to)
								: do_move<Black, Queen>(board, from, to);
		else
			return 2;
	}
	else
	{
		if (board.pawns & from)
			board.side == White ? do_move<White, Pawn>(board, from, to)
								: do_move<Black, Pawn>(board, from, to);
		else if (board.knights & from)
			board.side == White ? do_move<White, Knight>(board, from, to)
								: do_move<Black, Knight>(board, from, to);
		else if (board.bishops_queens & from)
			board.side == White ? do_move<White, Bishop>(board, from, to)
								: do_move<Black, Bishop>(board, from, to);
		else if (board.rooks_queens & from)
			board.side == White ? do_move<White, Rook>(board, from, to)
								: do_move<Black, Rook>(board, from, to);
		else if (board.white_king == from)
			do_move<White, King>(board, from, to);
		else if (board.black_king == from)
			do_move<Black, King>(board, from, to);
		else
			return 1;
	}

	return 0;
}

using Nodes = std::uint64_t;
using Depth = std::uint8_t;

template <Colour Us, PieceType T, bool Pinned, bool Divide = false>
inline Nodes perft_type(const Board &board, Bitboard pieces, const Bitboard targets, const Depth depth);

template <Colour Us, bool Divide = false>
inline Nodes perft_king(const Board &board, const Bitboard targets, const Depth depth);

template <Colour Us, bool Pinned, bool Divide = false>
inline Nodes perft_pawns(const Board &board, const Bitboard pawns, const Bitboard targets, const Depth depth);

template <Colour Us> inline Nodes count_moves(const Board &board);

template <Colour Us, bool Divide = false>
inline Nodes perft_colour(const Board &board, const Depth depth)
{
	if (depth == 0)
		return 1;

	if (!Divide && depth == 1)
		return count_moves<Us>(board);

	Nodes nodes = 0, cnt;

	const auto ksq      = Us == White ? board.white_king   : board.black_king;
	const auto friendly = Us == White ? board.white_pieces : board.black_pieces;
	const auto enemy    = Us == White ? board.black_pieces : board.white_pieces;

	const auto unsafe = unsafe_squares<Us>(board);

	auto targets = ~friendly;
	auto mask = friendly;

	nodes += perft_king<Us, Divide>(board, targets & ~unsafe, depth);

	// In check
	if (unsafe & ksq)
	{
		const auto checkers = checks<Us>(board);

		if (more_than_one(checkers))
			return nodes;

		const auto checker = static_cast<Square>(lsb(checkers));
		targets &= line_between(ksq, checker) | checkers;
	}
	else
	{
		// Short
		if (   (board.castling_rights.all & castling_rights(Us, true).all)
			&& !((friendly | enemy) & castling_rook_path(Us, true))
			&& !(unsafe & castling_king_path(Us, true)))
		{
			Board new_board = board;
			do_move<Us, King>(new_board, ksq, castling_king_dest(Us, true));
			cnt = perft_colour<~Us>(new_board, depth - 1);
			nodes += cnt;

			if (Divide)
				fmt::print("{}{}: {}\n", ksq, castling_king_dest(Us, true), cnt);
		}

		// Long
		if (   (board.castling_rights.all & castling_rights(Us, false).all)
			&& !((friendly | enemy) & castling_rook_path(Us, false))
			&& !(unsafe & castling_king_path(Us, false)))
		{
			Board new_board = board;
			do_move<Us, King>(new_board, ksq, castling_king_dest(Us, false));
			cnt = perft_colour<~Us>(new_board, depth - 1);
			nodes += cnt;

			if (Divide)
				fmt::print("{}{}: {}\n", ksq, castling_king_dest(Us, false), cnt);
		}
	}

	const auto pinned = pinned_pieces<Us>(board);

	nodes += perft_type<Us, Knight, false, Divide>(board, board.knights        & mask & ~pinned, targets, depth);
	nodes += perft_type<Us, Bishop, false, Divide>(board, board.bishops_queens & mask & ~pinned, targets, depth);
	nodes += perft_type<Us, Rook,   false, Divide>(board, board.rooks_queens   & mask & ~pinned, targets, depth);
	nodes += perft_pawns<Us, false, Divide>       (board, board.pawns          & mask & ~pinned, targets, depth);

	if (!(unsafe & ksq))
	{
		nodes += perft_type<Us, Bishop, true, Divide>(board, board.bishops_queens & mask & pinned, targets, depth);
		nodes += perft_type<Us, Rook,   true, Divide>(board, board.rooks_queens   & mask & pinned, targets, depth);
		nodes += perft_pawns<Us, true, Divide>       (board, board.pawns          & mask & pinned, targets, depth);
	}

	return nodes;
}

template <Colour Us, PieceType T, bool Pinned, bool Divide>
inline Nodes perft_type(const Board &board, Bitboard pieces, const Bitboard targets, const Depth depth)
{
	static_assert(T != King && T != Pawn, "Use count_king_moves/count_pawn_moves instead");

	Nodes nodes = 0, cnt;
	Board new_board;

	const auto ksq = Us == White ? board.white_king : board.black_king;
	const auto occ = board.white_pieces | board.black_pieces;

	while (pieces)
	{
		const auto from = static_cast<Square>(lsb(pieces));
		auto attacks = attacks_from<T>(from, occ) & targets;

		while (attacks)
		{
			const auto to = static_cast<Square>(lsb(attacks));
			attacks &= (attacks - 1);

			if (Pinned && !aligned(ksq, from, to))
				continue;

			new_board = board;
			do_move<Us, T>(new_board, from, to);
			cnt = perft_colour<~Us>(new_board, depth - 1);
			nodes += cnt;

			if (Divide)
				fmt::print("{}{}: {}\n", from, to, cnt);
		}

		pieces &= (pieces - 1);
	}

	return nodes;
}

template <Colour Us, bool Divide>
inline Nodes perft_king(const Board &board, const Bitboard targets, const Depth depth)
{
	Nodes nodes = 0, cnt;
	Board new_board;
	
	const auto ksq = Us == White ? board.white_king : board.black_king;

	auto attacks = attacks_from<King>(ksq) & targets;
	while (attacks)
	{
		const auto to = static_cast<Square>(lsb(attacks));
		attacks &= (attacks - 1);

		new_board = board;
		do_move<Us, King>(new_board, ksq, to);
		cnt = perft_colour<~Us>(new_board, depth - 1);
		nodes += cnt;

		if (Divide)
			fmt::print("{}{}: {}\n", ksq, to, cnt);
	}

	return nodes;
}

template <Colour Us, bool Divide = false>
inline Nodes perft_promotions(const Board &board, const Square from, const Square to, const Depth depth)
{
	Nodes nodes = 0, cnt;

	Board new_board = board;
	do_move<Us, Pawn, Knight>(new_board, from, to);
	cnt = perft_colour<~Us>(new_board, depth - 1);
	nodes += cnt;

	if (Divide)
		fmt::print("{}{}n: {}\n", from, to, cnt);

	new_board = board;
	do_move<Us, Pawn, Bishop>(new_board, from, to);
	cnt = perft_colour<~Us>(new_board, depth - 1);
	nodes += cnt;

	if (Divide)
		fmt::print("{}{}b: {}\n", from, to, cnt);

	new_board = board;
	do_move<Us, Pawn, Rook>(new_board, from, to);
	cnt = perft_colour<~Us>(new_board, depth - 1);
	nodes += cnt;

	if (Divide)
		fmt::print("{}{}r: {}\n", from, to, cnt);

	new_board = board;
	do_move<Us, Pawn, Queen>(new_board, from, to);
	cnt = perft_colour<~Us>(new_board, depth - 1);
	nodes += cnt;

	if (Divide)
		fmt::print("{}{}q: {}\n", from, to, cnt);

	return nodes;
}

template <Colour Us, bool Pinned, bool Divide>
inline Nodes perft_pawns(const Board &board, const Bitboard pawns, const Bitboard targets, const Depth depth)
{
	Nodes nodes = 0, cnt;
	Board new_board;

	constexpr auto Rank3  = Us == White ? Rank::Three : Rank::Six;
	constexpr auto Rank7  = Us == White ? Rank::Seven : Rank::Two;
	constexpr auto Up     = Us == White ? North       : South;
	constexpr auto UpWest = Up + West, UpEast = Up + East;

	const auto ksq   = Us == White ? board.white_king   : board.black_king;
	const auto enemy = Us == White ? board.black_pieces : board.white_pieces;
	const auto occ   = board.white_pieces | board.black_pieces, empty = ~occ;

	// En passant
	if (is_valid(board.en_passant))
	{
		if (const auto target = board.en_passant - Up; targets & target)
		{
			// 'candidates' is the bitboard of pawns that could perform en passant (at most two)
			auto candidates = pawn_attacks(~Us, board.en_passant) & pawns;
			while (candidates)
			{
				const auto from = static_cast<Square>(lsb(candidates));
				candidates &= (candidates - 1);

				// Check if performing en passant puts us in check.
				// Only sliding pieces can put us in check here.
				const auto new_occ = (occ ^ from ^ target) | board.en_passant;
				if (   (attacks_from<Bishop>(ksq, new_occ) & board.bishops_queens & enemy)
					|| (attacks_from<Rook>  (ksq, new_occ) & board.rooks_queens & enemy))
					continue;

				Board new_board = board;
				do_move<Us, Pawn>(new_board, from, board.en_passant);
				cnt = perft_colour<~Us>(new_board, depth - 1);
				nodes += cnt;

				if (Divide)
					fmt::print("{}{}: {}\n", from, board.en_passant, cnt);
			}
		}
	}

	const auto pawns_on_7 = pawns & Rank7, pawns_not_on_7 = pawns & ~pawns_on_7;

	// Pawn push, w/o promotion
	const auto single_push = shift<Up>(pawns_not_on_7) & empty;
	auto bb = single_push & targets;
	while (bb)
	{
		const auto to = static_cast<Square>(lsb(bb));
		const auto from = to - Up;
		bb &= (bb - 1);

		if (Pinned && !aligned(ksq, from, to))
			continue;

		new_board = board;
		do_move<Us, Pawn>(new_board, from, to);
		cnt = perft_colour<~Us>(new_board, depth - 1);
		nodes += cnt;

		if (Divide)
			fmt::print("{}{}: {}\n", from, to, cnt);
	}

	// Double pawn push
	bb = shift<Up>(single_push & Rank3) & empty & targets;
	while (bb)
	{
		const auto to = static_cast<Square>(lsb(bb)); 
		const auto from = to - Up * 2;
		bb &= (bb - 1);

		if (Pinned && !aligned(ksq, from, to))
			continue;

		new_board = board;
		do_move<Us, Pawn>(new_board, from, to);
		cnt = perft_colour<~Us>(new_board, depth - 1);
		nodes += cnt;

		if (Divide)
			fmt::print("{}{}: {}\n", from, to, cnt);
	}

	// Promotions, w/o captures
	if (!Pinned)
	{
		bb = shift<Up>(pawns_on_7) & empty & targets;
		while (bb)
		{
			const auto to = static_cast<Square>(lsb(bb));
			const auto from = to - Up;
			bb &= (bb - 1);

			nodes += perft_promotions<Us, Divide>(board, from, to, depth);
		}
	}

	// Captures, w/o promotion, 1/2
	bb = shift<UpWest>(pawns_not_on_7) & enemy & targets;
	while (bb)
	{
		const auto to = static_cast<Square>(lsb(bb));
		const auto from = to - UpWest;
		bb &= (bb - 1);
		
		if (Pinned && !aligned(ksq, from, to))
			continue;

		new_board = board;
		do_move<Us, Pawn>(new_board, from, to);
		cnt = perft_colour<~Us>(new_board, depth - 1);
		nodes += cnt;

		if (Divide)
			fmt::print("{}{}: {}\n", from, to, cnt);
	}

	// Captures, w/o promotion, 2/2
	bb = shift<UpEast>(pawns_not_on_7) & enemy & targets;
	while (bb)
	{
		const auto to = static_cast<Square>(lsb(bb));
		const auto from = to - UpEast;
		bb &= (bb - 1);
		
		if (Pinned && !aligned(ksq, from, to))
			continue;

		new_board = board;
		do_move<Us, Pawn>(new_board, from, to);
		cnt = perft_colour<~Us>(new_board, depth - 1);
		nodes += cnt;

		if (Divide)
			fmt::print("{}{}: {}\n", from, to, cnt);
	}

	// Captures, w/ promotion, 1/2
	bb = shift<UpWest>(pawns_on_7) & enemy & targets;
	while (bb)
	{
		const auto to = static_cast<Square>(lsb(bb));
		const auto from = to - UpWest;
		bb &= (bb - 1);
		
		if (Pinned && !aligned(ksq, from, to))
			continue;

		nodes += perft_promotions<Us, Divide>(board, from, to, depth);
	}

	// Captures, w/ promotion, 2/2
	bb = shift<UpEast>(pawns_on_7) & enemy & targets;
	while (bb)
	{
		const auto to = static_cast<Square>(lsb(bb));
		const auto from = to - UpEast;
		bb &= (bb - 1);
		
		if (Pinned && !aligned(ksq, from, to))
			continue;

		nodes += perft_promotions<Us, Divide>(board, from, to, depth);
	}

	return nodes;
}

//
// Specialised functions for counting at leaf nodes
//

template <Colour Us, PieceType T, bool Pinned>
inline Nodes count_type(const Board &board, Bitboard pieces, const Bitboard targets);

template <Colour Us, bool Pinned>
inline Nodes count_pawns(const Board &board, const Bitboard pawns, const Bitboard targets);

template <Colour Us>
inline Nodes count_moves(const Board &board)
{
	Nodes nodes = 0, cnt;

	const auto ksq      = Us == White ? board.white_king   : board.black_king;
	const auto friendly = Us == White ? board.white_pieces : board.black_pieces;
	const auto enemy    = Us == White ? board.black_pieces : board.white_pieces;

	const auto unsafe = unsafe_squares<Us>(board);

	auto targets = ~friendly;
	auto mask = friendly;

	nodes += popcount(attacks_from<King>(ksq) & targets & ~unsafe);

	// In check
	if (unsafe & ksq)
	{
		const auto checkers = checks<Us>(board);

		if (more_than_one(checkers))
			return nodes;

		const auto checker = static_cast<Square>(lsb(checkers));
		targets &= line_between(ksq, checker) | checkers;
	}
	else
	{
		// Short
		if (   (board.castling_rights.all & castling_rights(Us, true).all)
			&& !((friendly | enemy) & castling_rook_path(Us, true))
			&& !(unsafe & castling_king_path(Us, true)))
			++nodes;

		// Long
		if (   (board.castling_rights.all & castling_rights(Us, false).all)
			&& !((friendly | enemy) & castling_rook_path(Us, false))
			&& !(unsafe & castling_king_path(Us, false)))
			++nodes;
	}

	const auto pinned = pinned_pieces<Us>(board);

	nodes += count_type<Us, Knight, false>(board, board.knights        & mask & ~pinned, targets);
	nodes += count_type<Us, Bishop, false>(board, board.bishops_queens & mask & ~pinned, targets);
	nodes += count_type<Us, Rook,   false>(board, board.rooks_queens   & mask & ~pinned, targets);
	nodes += count_pawns<Us, false>       (board, board.pawns          & mask & ~pinned, targets);

	if (!(unsafe & ksq))
	{
		nodes += count_type<Us, Bishop, true>(board, board.bishops_queens & mask & pinned, targets);
		nodes += count_type<Us, Rook,   true>(board, board.rooks_queens   & mask & pinned, targets);
		nodes += count_pawns<Us, true>       (board, board.pawns          & mask & pinned, targets);
	}

	return nodes;
}

template <Colour Us, PieceType T, bool Pinned>
inline Nodes count_type(const Board &board, Bitboard pieces, const Bitboard targets)
{
	static_assert(T != King && T != Pawn, "Use count_king_moves/count_pawn_moves instead");

	Nodes nodes = 0;

	const auto ksq = Us == White ? board.white_king : board.black_king;
	const auto occ = board.white_pieces | board.black_pieces;

	while (pieces)
	{
		const auto from = static_cast<Square>(lsb(pieces));
		auto attacks = attacks_from<T>(from, occ) & targets;

		if (!Pinned)
			nodes += popcount(attacks);
		else
		{
			while (attacks)
			{
				const auto to = static_cast<Square>(lsb(attacks));
				attacks &= (attacks - 1);

				if (!aligned(ksq, from, to))
					continue;

				++nodes;
			}
		}

		pieces &= (pieces - 1);
	}

	return nodes;
}

template <Colour Us, bool Pinned>
inline Nodes count_pawns(const Board &board, const Bitboard pawns, const Bitboard targets)
{
	Nodes nodes = 0;

	constexpr auto Rank3  = Us == White ? Rank::Three : Rank::Six;
	constexpr auto Rank7  = Us == White ? Rank::Seven : Rank::Two;
	constexpr auto Up     = Us == White ? North       : South;
	constexpr auto UpWest = Up + West, UpEast = Up + East;

	const auto ksq   = Us == White ? board.white_king   : board.black_king;
	const auto enemy = Us == White ? board.black_pieces : board.white_pieces;
	const auto occ   = board.white_pieces | board.black_pieces, empty = ~occ;

	// En passant
	if (is_valid(board.en_passant))
	{
		if (const auto target = board.en_passant - Up; targets & target)
		{
			// 'candidates' is the bitboard of pawns that could perform en passant (at most two)
			auto candidates = pawn_attacks(~Us, board.en_passant) & pawns;
			while (candidates)
			{
				const auto from = static_cast<Square>(lsb(candidates));
				candidates &= (candidates - 1);

				// Check if performing en passant puts us in check.
				// Only sliding pieces can put us in check here.
				const auto new_occ = (occ ^ from ^ target) | board.en_passant;
				if (   (attacks_from<Bishop>(ksq, new_occ) & board.bishops_queens & enemy)
					|| (attacks_from<Rook>  (ksq, new_occ) & board.rooks_queens & enemy))
					continue;

				++nodes;
			}
		}
	}

	const auto pawns_on_7 = pawns & Rank7, pawns_not_on_7 = pawns & ~pawns_on_7;

	// Pawn push, w/o promotion
	const auto single_push = shift<Up>(pawns_not_on_7) & empty;
	auto bb = single_push & targets;

	if (!Pinned)
		nodes += popcount(bb);
	else
	{
		while (bb)
		{
			const auto to = static_cast<Square>(lsb(bb));
			const auto from = to - Up;
			bb &= (bb - 1);

			if (!aligned(ksq, from, to))
				continue;

			++nodes;
		}
	}

	// Double pawn push
	bb = shift<Up>(single_push & Rank3) & empty & targets;

	if (!Pinned)
		nodes += popcount(bb);
	else
	{
		while (bb)
		{
			const auto to = static_cast<Square>(lsb(bb)); 
			const auto from = to - Up * 2;
			bb &= (bb - 1);

			if (!aligned(ksq, from, to))
				continue;

			++nodes;
		}
	}

	// Promotions, w/o captures
	if (!Pinned)
	{
		bb = shift<Up>(pawns_on_7) & empty & targets;
		nodes += popcount(bb) * 4;
	}

	// Captures, w/o promotion, 1/2
	bb = shift<UpWest>(pawns_not_on_7) & enemy & targets;
	if (!Pinned)
		nodes += popcount(bb);
	else
	{
		while (bb)
		{
			const auto to = static_cast<Square>(lsb(bb));
			const auto from = to - UpWest;
			bb &= (bb - 1);
			
			if (!aligned(ksq, from, to))
				continue;

			++nodes;
		}
	}

	// Captures, w/o promotion, 2/2
	bb = shift<UpEast>(pawns_not_on_7) & enemy & targets;
	if (!Pinned)
		nodes += popcount(bb);
	else
	{
		while (bb)
		{
			const auto to = static_cast<Square>(lsb(bb));
			const auto from = to - UpEast;
			bb &= (bb - 1);
			
			if (!aligned(ksq, from, to))
				continue;

			++nodes;
		}
	}

	// Captures, w/ promotion, 1/2
	bb = shift<UpWest>(pawns_on_7) & enemy & targets;
	if (!Pinned)
		nodes += popcount(bb) * 4;
	else
	{
		while (bb)
		{
			const auto to = static_cast<Square>(lsb(bb));
			const auto from = to - UpWest;
			bb &= (bb - 1);
			
			if (!aligned(ksq, from, to))
				continue;

			nodes += 4;
		}
	}

	// Captures, w/ promotion, 2/2
	bb = shift<UpEast>(pawns_on_7) & enemy & targets;
	if (!Pinned)
		nodes += popcount(bb) * 4;
	else
	{
		while (bb)
		{
			const auto to = static_cast<Square>(lsb(bb));
			const auto from = to - UpEast;
			bb &= (bb - 1);
			
			if (!aligned(ksq, from, to))
				continue;

			nodes += 4;
		}
	}

	return nodes;
}
