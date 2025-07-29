#![allow(
    non_snake_case,
    non_camel_case_types,
    non_upper_case_globals,
    dead_code,
    unused_labels,
    unused_macros
)]

const INFO_FLAGS: isize = 0 * WORD;
const INFO_DEPTH: isize = 1 * WORD;
const INFO_LHS_RS: isize = 2 * WORD;
const INFO_LHS_CS: isize = 3 * WORD;
const INFO_RHS_RS: isize = 4 * WORD;
const INFO_RHS_CS: isize = 5 * WORD;
const INFO_ALPHA: isize = 6 * WORD;
const INFO_PTR: isize = 7 * WORD;
const INFO_RS: isize = 8 * WORD;
const INFO_CS: isize = 9 * WORD;
const INFO_ROW_IDX: isize = 10 * WORD;
const INFO_COL_IDX: isize = 11 * WORD;
const INFO_DIAG_PTR: isize = 12 * WORD;
const INFO_DIAG_STRIDE: isize = 13 * WORD;

use std::env;
use std::fmt::Display;
use std::fs;
use std::path::Path;
use std::sync::LazyLock;

use std::cell::Cell;
use std::cell::RefCell;
use std::ops::Index;

type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

macro_rules! setup {
    ($ctx: ident $(,)?) => {
        macro_rules! ctx {
            () => {
                $ctx
            };
        }
    };

    ($ctx: ident, $target: ident $(,)?) => {
        macro_rules! ctx {
            () => {
                $ctx
            };
        }
        macro_rules! target {
            () => {
                $target
            };
        }
    };
}

macro_rules! align {
    () => {
        asm!(".align 16")
    };
}

macro_rules! func {
    ($name: tt) => {
        let __name__ = &format!($name);

        asm!(".globl {__name__}");
        align!();
        asm!("{__name__}:");
        defer!(asm!("ret"));

        macro_rules! name {
            () => {
                __name__
            };
        }
    };
}

macro_rules! asm {
    ($code: tt) => {{
        asm!($code, "");
    }};

    ($code: tt, $comment: tt) => {{
        use std::fmt::Write;

        let code = &mut *ctx!().code.borrow_mut();

        ::interpol::writeln!(code, $code).unwrap();
    }};
}

macro_rules! reg {
    ($name: ident) => {
        let $name = ctx!().reg(::std::stringify!($name));
        ::defer::defer!(ctx!().reg_drop($name, ::std::stringify!($name)));
    };

    (&$name: ident) => {
        $name = ctx!().reg(::std::stringify!($name));
        ::defer::defer!(ctx!().reg_drop($name, ::std::stringify!($name)));
    };
}

macro_rules! label {
    ({$(let $label: ident;)*}) => {$(
        let $label = Cell::new(!ctx!().label(""));
        defer!({
            let mut __label__ = $label.get();
            if (__label__ as isize) < 0 {
                __label__ = !__label__;
            }
            ctx!().label_drop(__label__, "");
        });
    )*};

    ($label: ident) => {{
        let __label__ = $label.get();
        if __label__ as isize >= 0 {
            format!("{__label__}b")
        } else {
            format!("{!__label__}f")
        }
    }};

    ($label: ident = _) => {{
        let __label__ = !$label.get();
        assert!(__label__ as isize > 0);
        $label.set(__label__);

        align!();
        asm!("{__label__}:");
    }};
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Reg {
    rax = 0,
    rbx = 1,
    rcx = 2,
    rdx = 3,
    rdi = 4,
    rsi = 5,
    rbp = 6,
    rsp = 7,
    r8 = 8,
    r9 = 9,
    r10 = 10,
    r11 = 11,
    r12 = 12,
    r13 = 13,
    r14 = 14,
    r15 = 15,
    rip = 16,
}


type Code = RefCell<String>;

struct Ctx {
    reg_busy: [Cell<bool>; 16],
    label: Cell<usize>,
    code: Code,
}

impl Ctx {
    fn new() -> Self {
        Self {
            reg_busy: [const { Cell::new(false) }; 16],
            label: Cell::new(200000),
            code: RefCell::new(String::new()),
        }
    }

    #[track_caller]
    fn reg(&self, _: &str) -> Reg {
        setup!(self);

        for &reg in Reg::ALL {
            if !self[reg].get() {
                // We'll use a placeholder for now since we removed the asm! macro
                self[reg].set(true);
                return reg;
            }
        }

        panic!();
    }

    fn reg_drop(&self, reg: Reg, _: &str) {
        self[reg].set(false);
        // Placeholder for asm!("pop {reg}");
    }

    fn label(&self, _: &str) -> usize {
        let label = self.label.get();
        self.label.set(label + 1);
        label
    }

    fn label_drop(&self, label: usize, _: &str) {
        self.label.set(label);
    }
}

impl Reg {
    const ALL: &[Self] = &[
        Self::rax,
        Self::rbx,
        Self::rcx,
        Self::rdx,
        Self::rdi,
        Self::rsi,
        Self::rbp,
        Self::rsp,
        Self::r8,
        Self::r9,
        Self::r10,
        Self::r11,
        Self::r12,
        Self::r13,
        Self::r14,
        Self::r15,
    ];
}

impl std::fmt::Display for Reg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

impl Index<Reg> for Ctx {
    type Output = Cell<bool>;

    fn index(&self, index: Reg) -> &Self::Output {
        &self.reg_busy[index as usize]
    }
}

const VERSION_MAJOR: usize = 0;

const PRETTY: LazyLock<bool> = LazyLock::new(|| false);
const PREFIX: LazyLock<String> = LazyLock::new(|| {
    if *PRETTY {
        format!("[gemm.x86 v{VERSION_MAJOR}]")
    } else {
        format!("gemm_v{VERSION_MAJOR}")
    }
});
const WORD: isize = 8;
const QUOTE: char = '"';

fn func_name(pieces: &str, params: &str, quote: bool) -> String {
    let pieces = pieces.split('.').collect::<Vec<_>>();
    let params = params
        .split('.')
        .filter_map(|p| {
            if p.is_empty() {
                return None;
            }

            let mut iter = p.split('=');
            Some((iter.next().unwrap(), iter.next().unwrap()))
        })
        .collect::<Vec<_>>();

    if *PRETTY {
        let name = pieces
            .iter()
            .map(|i| i.as_ref())
            .collect::<Vec<_>>()
            .join(".");
        let params = params
            .iter()
            .map(|(k, v)| format!("{k} = {v}"))
            .collect::<Vec<_>>()
            .join(", ");

        if params.is_empty() {
            format!("{}{} {}{}", QUOTE, *PREFIX, name, QUOTE)
        } else {
            format!("{}{} {} [with {}]{}", QUOTE, *PREFIX, name, params, QUOTE)
        }
    } else {
        let name = pieces
            .iter()
            .map(|i| i.as_ref())
            .collect::<Vec<_>>()
            .join("_");
        let params = params
            .iter()
            .map(|(k, v)| format!("{k}{v}"))
            .collect::<Vec<_>>()
            .join("_");

        let name = if params.is_empty() {
            format!("{}_{}", *PREFIX, name)
        } else {
            format!("{}_{}_{}", *PREFIX, name, params)
        };

        if quote {
            format!("{QUOTE}{name}{QUOTE}")
        } else {
            format!("{name}")
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Ty {
    F32,
    F64,
    C32,
    C64,
}

impl Ty {
    fn sizeof(self) -> isize {
        match self {
            Ty::F32 => 4,
            Ty::F64 => 8,
            Ty::C32 => 2 * 4,
            Ty::C64 => 2 * 8,
        }
    }

    fn suffix(self) -> String {
        match self {
            Ty::F32 | Ty::C32 => "s",
            Ty::F64 | Ty::C64 => "d",
        }
        .to_string()
    }
}

impl Display for Ty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match *self {
            Ty::F32 => "f32",
            Ty::F64 => "f64",
            Ty::C32 => "c32",
            Ty::C64 => "c64",
        })
    }
}

fn main() -> Result {
    // Generate some basic function stubs for now
    let mut code = String::new();

    // Create empty function arrays for each type
    let mut f32_simd256 = vec![];
    let mut f32_simd512x4 = vec![];
    let mut f64_simd256 = vec![];
    let mut f64_simd512x4 = vec![];
    let mut f64_simd512x8 = vec![];
    let mut c32_simd256 = vec![];
    let mut c32_simd512x4 = vec![];
    let mut c64_simd256 = vec![];
    let mut c64_simd512x4 = vec![];

    let mut f32_simdpack_256 = vec![];
    let mut f32_simdpack_512 = vec![];
    let mut f64_simdpack_256 = vec![];
    let mut f64_simdpack_512 = vec![];
    let mut c32_simdpack_256 = vec![];
    let mut c32_simdpack_512 = vec![];
    let mut c64_simdpack_256 = vec![];
    let mut c64_simdpack_512 = vec![];

    // Generate simple stub functions for now
    for ty in [Ty::F32, Ty::F64, Ty::C32, Ty::C64] {
        for (simd, mr_nr_list) in [
            ("256", vec![(24, 4), (12, 4), (12, 4), (6, 4)]),
            ("512x4", vec![(96, 4), (48, 4), (48, 4), (24, 4)]),
            ("512x8", vec![(0, 0), (24, 8), (0, 0), (0, 0)]),
        ] {
            let (mr, nr) = mr_nr_list[match ty {
                Ty::F32 => 0,
                Ty::F64 => 1,
                Ty::C32 => 2,
                Ty::C64 => 3,
            }];

            if mr == 0 || nr == 0 {
                continue;
            }

            // Generate microkernel functions
            for n in 1..=nr {
                let name = func_name(
                    &format!("gemm.microkernel.{ty}.simd{simd}"),
                    &format!("m={mr}.n={n}"),
                    false,
                );
                
                code += &format!(
                    ".globl {name}\n.align 16\n{name}:\nret\n\n"
                );

                match (ty, simd) {
                    (Ty::F32, "256") => f32_simd256.push(name),
                    (Ty::F32, "512x4") => f32_simd512x4.push(name),
                    (Ty::F64, "256") => f64_simd256.push(name),
                    (Ty::F64, "512x4") => f64_simd512x4.push(name),
                    (Ty::F64, "512x8") => f64_simd512x8.push(name),
                    (Ty::C32, "256") => c32_simd256.push(name),
                    (Ty::C32, "512x4") => c32_simd512x4.push(name),
                    (Ty::C64, "256") => c64_simd256.push(name),
                    (Ty::C64, "512x4") => c64_simd512x4.push(name),
                    _ => {}
                }
            }

            // Generate pack functions
            let pack_name = func_name(
                &format!("gemm.pack.{ty}.simd{simd}"),
                &format!("m={mr}"),
                false,
            );
            
            code += &format!(
                ".globl {pack_name}\n.align 16\n{pack_name}:\nret\n\n"
            );

            match (ty, simd) {
                (Ty::F32, "256") => f32_simdpack_256.push(pack_name),
                (Ty::F32, "512x4") => f32_simdpack_512.push(pack_name),
                (Ty::F64, "256") => f64_simdpack_256.push(pack_name),
                (Ty::F64, "512x4") => f64_simdpack_512.push(pack_name),
                (Ty::C32, "256") => c32_simdpack_256.push(pack_name),
                (Ty::C32, "512x4") => c32_simdpack_512.push(pack_name),
                (Ty::C64, "256") => c64_simdpack_256.push(pack_name),
                (Ty::C64, "512x4") => c64_simdpack_512.push(pack_name),
                _ => {}
            }
        }
    }

    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("asm.s");
    fs::write(&dest_path, &code)?;

    // Generate Rust code
    {
        let dest_path = Path::new(&out_dir).join("asm.rs");

        let mut code = format!(
            "::core::arch::global_asm!{{
include_str!(concat!(env!({}OUT_DIR{}), {}/asm.s{})) }}",
            QUOTE, QUOTE, QUOTE, QUOTE
        );

        for (names, ty, bits) in [
            (&f32_simd256, Ty::F32, "256"),
            (&f32_simd512x4, Ty::F32, "512x4"),
            (&f64_simd256, Ty::F64, "256"),
            (&f64_simd512x4, Ty::F64, "512x4"),
            (&f64_simd512x8, Ty::F64, "512x8"),
            (&c32_simd256, Ty::C32, "256"),
            (&c32_simd512x4, Ty::C32, "512x4"),
            (&c64_simd256, Ty::C64, "256"),
            (&c64_simd512x4, Ty::C64, "512x4"),
            (&f32_simdpack_256, Ty::F32, "pack_256"),
            (&f32_simdpack_512, Ty::F32, "pack_512"),
            (&f64_simdpack_256, Ty::F64, "pack_256"),
            (&f64_simdpack_512, Ty::F64, "pack_512"),
            (&c32_simdpack_256, Ty::C32, "pack_256"),
            (&c32_simdpack_512, Ty::C32, "pack_512"),
            (&c64_simdpack_256, Ty::C64, "pack_256"),
            (&c64_simdpack_512, Ty::C64, "pack_512"),
        ] {
            for (i, name) in names.iter().enumerate() {
                let name = if name.starts_with('"') {
                    name.clone()
                } else {
                    format!("{}{}{}", QUOTE, name, QUOTE)
                };
                code += &format!(
                    "
                unsafe extern {}C{} {{
                    #[link_name = {name}]
                    unsafe fn __decl_{ty}_simd{bits}_{i}__();
                }}
                ",
                    QUOTE, QUOTE
                );
            }

            let upper = format!("{ty}").to_uppercase();
            let names_len = names.len();
            code += &format!(
                "pub static {upper}_SIMD{bits}: [unsafe extern {}C{} fn(); {names_len}] = [",
                QUOTE, QUOTE
            );
            for i in 0..names.len() {
                code += &format!("__decl_{ty}_simd{bits}_{i}__,");
            }
            code += "];";
        }

        fs::write(&dest_path, &code)?;
    }

    println!("cargo::rerun-if-changed=build.rs");

    Ok(())
}
