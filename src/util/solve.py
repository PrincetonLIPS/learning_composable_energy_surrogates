import math
import copy


def solve(
    fem,
    args,
    q,
    guess,
    q_last,
    max_iter,
    factor,
    recursion_depth=0,
):
    try:
        # print("recursion {}, iter {}, factor {}".format(recursion_depth, max_iter, factor))
        new_args = copy.deepcopy(args)
        new_args = new_args
        new_args.max_newton_iter = max_iter
        new_args.relaxation_parameter = factor
        T = 2 if recursion_depth == 0 else 10
        Z = sum([2 ** i for i in range(T)])
        new_guess = guess
        for i in range(T):
            new_args.atol = (
                args.atol
            )  # 10**(math.log10(args.atol)*2**i / (2**(T-1)))
            new_args.rtol = 10 ** (math.log10(args.rtol) * 2 ** i / Z)
            new_args.max_newton_iter = int(math.ceil(2 ** i * max_iter / Z)) + 1
            # print("solve with rtol {} atol {} iter {} factor {} u_norm {} guess_norm {}".format(
            #    new_args.atol, new_args.rtol, new_args.max_newton_iter, new_args.relaxation_parameter, q.norm().item(),
            #    torch.Tensor(guess).norm().item()))
            f, u = fem.f(q, initial_guess=new_guess, return_u=True, args=new_args)
            new_guess = u.vector()
        # print("energy: {:.3e}, sq(q): {:.3e},  f/sq(q): {:.3e}".format(f, sq(q), (f+EPS)/sq(q)))
        return u.vector()
    except Exception as e:
        if q_last is None and recursion_depth == 0:
            return solve(fem, args, q, guess, q_last, max_iter, factor=0.1, recursion_depth=1)
        elif q_last is None:
            raise e
        elif recursion_depth >= 8:
            # print("Maximum recursion depth exceeded! giving up.")
            raise e
        else:
            # print("recursing due to error, depth {}:".format(recursion_depth+1))
            # print(e)
            # q_mid = q_last + 0.5*(q-q_last)
            new_factor = 0.1 # max(factor*0.5, 0.05)
            new_max_iter = int(
                5
                + max_iter
                * math.log(1.0 - min(0.9, factor))
                / math.log(1.0 - new_factor)
            )
            # print("new factor {}, new max iter {}".format(new_factor, new_max_iter))

            # guess = solve(q_mid, guess, q_last, max_iter=new_max_iter,
            #               factor=new_factor, recursion_depth=recursion_depth+1)
            # print("first half of recursion {}".format(recursion_depth+1))
            guess = solve(fem, args,
                (q+q_last)/2,
                guess,
                q_last,
                max_iter=new_max_iter,
                factor=new_factor,
                recursion_depth=recursion_depth + 1,
            )
            # print("second half of recursion {}".format(recursion_depth+1))
            return solve(fem, args,
                q,
                guess,
                (q+q_last)/2,
                max_iter=new_max_iter,
                factor=new_factor,
                recursion_depth=recursion_depth + 1,
            )
