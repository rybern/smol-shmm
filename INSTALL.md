This package uses the build tool [Stack](https://docs.haskellstack.org/en/stable/README/#how-to-install).

It also has some additional library dependencies. I use Stack's [Nix](https://nixos.org/nix/) integration to manage my libraries. If you have Nix installed, you can use stack.yaml that comes with this package to automatically install the dependencies. If not, you can install them with another package manager. I think the following list is exhaustive:

Required:
 - eigen 3.3.4
 - gcc

Optional:
 - ihaskell (for the Jupyter notebook)
 - cairo (for plotting)

To build without Nix, run `stack build`. To build with Nix, run `stack build --nix`. To install globally on your machine, replace `build` with `install`.

To open a Haskell REPL with SMoL-SHMM loaded automatically, run `stack ghci`.
