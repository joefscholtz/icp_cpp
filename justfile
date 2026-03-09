alias b := build
alias re:=rebuild
alias r := run

default:
  just --list

install:
  @echo "not implemented"

rebuild: && build
  rm -rf build

configure:
  cmake --preset default

build: configure
  cmake --build --preset default --parallel
  ln -sf build/compile_commands.json .

run:
  ./build/app/icp_app
