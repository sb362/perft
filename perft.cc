#include "perft.hh"

#include "cxxopts.hh"

#include <chrono>

using Microseconds = std::chrono::microseconds;
using Milliseconds = std::chrono::milliseconds;
using Clock = std::chrono::high_resolution_clock;
using std::chrono::duration_cast;

#if defined(NDEBUG)
constexpr bool IncreaseDepth = true;
#else
constexpr bool IncreaseDepth = false;
#endif

struct NameFENDepth
{
	std::string name, fen;
	Depth depth;
};

static std::array<NameFENDepth, 7> PredefinedFENs
{{
	{"startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",
		IncreaseDepth ? 7 : 5},
	{"kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -",
		IncreaseDepth ? 6 : 5},
	{"pins", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -",
		IncreaseDepth ? 8 : 6},
	{"cpw4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq -",
		IncreaseDepth ? 6 : 5},
	{"cpw5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ -",
		IncreaseDepth ? 6 : 5},
	{"cpw6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - -",
		IncreaseDepth ? 6 : 5},
	{"promotions", "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - -",
		IncreaseDepth ? 7 : 6}
}};

template <bool Divide = false> Nodes perft(const Board &board, const Depth depth)
{
	return board.side == White ? perft_colour<White, Divide>(board, depth)
							   : perft_colour<Black, Divide>(board, depth);
}

std::string compiler_info();

int main(int argc, char *argv[])
{
	cxxopts::Options options("Perft", "Ultra-fast perft calculator");
	options.add_options()
		("f,fen", "FEN string", cxxopts::value<std::string>())
		("m,moves", "Comma-separated list of moves in UCI form to apply to the root position",
					cxxopts::value<std::vector<std::string>>())
		("d,depth", "Depth", cxxopts::value<unsigned>())
		("u,upto", "Calculate for depths 1...n")
		("b,bench", "Benchmark mode")
		("divide", "Print move counts for each root move")
		("c,compiler", "Show compiler info");

	auto result = options.parse(argc, argv);

	if (result["compiler"].as<bool>())
		fmt::print("{}\n", compiler_info());

	//unsigned threads =
	//	util::clamp(result["threads"].as<unsigned>(), 1u, std::thread::hardware_concurrency());

	bool upto = result["upto"].as<bool>();
	bool bench = result["bench"].as<bool>();
	bool divide = result["divide"].as<bool>();
	bool verify = false/*result.count("verify")*/;

	if ((bench && (divide || upto || verify)) || (upto && (divide || bench || verify)) ||
		(verify && (divide || bench || upto)) || (divide && (bench || upto || verify)))
	{
		fmt::print("Incorrect usage: bench, divide, upto, verify are mutually exclusive options\n");
		return 0;
	}

	Depth depth = result.count("depth") ? result["depth"].as<unsigned>() : 0;

	if (result.count("fen"))
	{
		std::string fen = result["fen"].as<std::string>();

		for (const auto &name_fen_depth : PredefinedFENs)
		{
			if (name_fen_depth.name == fen)
				fen = name_fen_depth.fen;

			if (name_fen_depth.fen == fen)
			{
				if (depth == 0)
					depth = name_fen_depth.depth;

				break;
			}
		}

		if (depth == 0)
		{
			fmt::print("Error: depth is zero\n");
			return 0;
		}

		Board board;
		if (int status = parse_fen(board, fen); status == 0)
		{
			if (result.count("moves"))
			{
				const auto moves = result["moves"].as<std::vector<std::string>>();
				for (const auto &move : moves)
					if (status = parse_and_push_uci(board, move); status != 0)
					{
						fmt::print(
							"Error: move parser returned non-zero code {} when parsing '{}'\n",
							status, move);
						return 0;
					}
			}

			fmt::print("{}\n", to_string(board));

			if (!divide)
			{
				fmt::print("{: <6} {: <12} {: <12} {}\n", "Depth", "Nodes", "Time (ms)",
						   "Nodes/sec");
			}

			Nodes nodes;
			for (Depth d = (upto ? 1 : depth); d <= depth; ++d)
			{
				const auto t0 = Clock::now();
				nodes = divide ? perft<true>(board, d) : perft<false>(board, d);
				const auto t1 = Clock::now();
				const auto dt = duration_cast<Microseconds>(t1 - t0);

				if (divide)
				{
					fmt::print("\n{} nodes\n{} ms\n{:.0f} nodes/sec\n", nodes,
							   duration_cast<Milliseconds>(dt).count(), (1e6 * nodes) / dt.count());
				}
				else
				{
					fmt::print("{: <6} {: <12} {: <12} {:.0f}\n", d, nodes,
							   duration_cast<Milliseconds>(dt).count(), (1e6 * nodes) / dt.count());
				}
			}
		}
		else
		{
			fmt::print("Error: FEN parser returned non-zero code {} when parsing '{}'\n", status,
					   fen);

			return 0;
		}
	}
	else if (bench)
	{
		fmt::print("{: <10} {: <6} {: <12} {: <12} {}\n", "Name", "Depth", "Nodes", "Time (ms)",
				   "Nodes/sec");

		Nodes total_nodes = 0, nodes;
		Milliseconds total_time {};
		for (const auto &name_fen_depth : PredefinedFENs)
		{
			Board board;
			if (const auto status = parse_fen(board, name_fen_depth.fen); status != 0)
			{
				fmt::print("Error: FEN parser returned non-zero code {} when parsing '{}' ({})\n",
						   status, name_fen_depth.fen);
				break;
			}

			const auto t0 = Clock::now();
			nodes = perft(board, name_fen_depth.depth);
			const auto t1 = Clock::now();
			const auto dt = duration_cast<Microseconds>(t1 - t0);

			total_nodes += nodes;
			total_time += duration_cast<Milliseconds>(dt);

			fmt::print("{: <10} {: <6} {: <12} {: <12} {:.0f}\n", name_fen_depth.name,
					   name_fen_depth.depth, nodes, duration_cast<Milliseconds>(dt).count(),
					   (1e6 * nodes) / dt.count());
		}

		fmt::print("{: <10} {: <6} {: <12} {: <12} {:.0f}\n", "total/avg", '-', total_nodes,
				   total_time.count(), (1e3 * total_nodes) / total_time.count());
	}
	else if (verify)
	{
		fmt::print("Not implemented.\n");
	}
	else
	{
		// Incorrect usage
		fmt::print("{}\n{}\n", options.help(), "Predefined FENs:");
		for (const auto &name_fen_depth : PredefinedFENs)
			fmt::print(" {: <10} {}\n", name_fen_depth.name, name_fen_depth.fen);
	}

	return 0;
}

inline std::string compiler_info()
{
	std::string out;

	out += "OS: ";

#if defined(__linux__)
	out += "Linux\n";
#elif defined(_WIN64)
	out += "Windows\n";
#elif defined(__APPLE__)
	out += "Apple\n"
#elif defined(__MINGW64__)
	out += "MinGW\n";
#elif defined(__CYGWIN__)
	out += "Cygwin\n";
#else
	out += "unknown\n";
#endif

	out += "Compiler: ";

#if defined(__clang__)
	out += fmt::format("Clang {}.{}.{}\n", __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(__GNUC__)
	out += fmt::format("GCC {}.{}.{}\n", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#elif defined(__INTEL_COMPILER)
	out += fmt::format("ICC (v {}.{})\n", __INTEL_COMPILER, __INTEL_COMPILER_UPDATE);
#elif defined(__MSC_VER)
	out += fmt::format("MSVC (v {})\n", __MSC_VER);
#else
	out += "unknown\n";
#endif

#if !defined(NDEBUG)
	out += "Debug\n";
#endif

	if constexpr (HasLsbIntrinsics)
		out += "LSB intrinsics\n";

	if constexpr (HasPopcntIntrinsics)
		out += "POPCNT intrinsics\n";

	if constexpr (HasBMI2)
		out += "BMI2 intrinsics\n";

	out += "Move generation: ";

#if defined(USE_KOGGE)
	out += "Kogge-Stone\n";
#elif defined(USE_FANCY)
	out += "fancy magic bitboards\n";
#elif defined(USE_PEXT)
	out += "PEXT bitboards\n";
#elif defined(USE_PDEP)
	out += "PEXT+PDEP bitboards\n";
#else
	out += "unknown\n";
#endif

	return out;
}
