kpse.set_program_name("luatex")

-- file.nameonly comes from ConTeXt lualibs, not available in standalone texlua
file = file or {}
file.nameonly = file.nameonly or function(name)
  local basename = name:match("([^/]+)$") or name
  return basename:match("(.+)%.[^.]+$") or basename
end

-- Load newpax.lua and patch nil-unsafe data[N][2] accesses in outputENTRY_dest.
-- Chrome PDFs use direct destinations with null coords (/XYZ null null null),
-- causing data[3..5] to be nil. The original code does "if data[N][2] then"
-- which crashes; we add a "data[N] and" guard.
local path = kpse.find_file("newpax.lua", "lua")
local f = io.open(path, "r")
local code = f:read("*a")
f:close()

code = code:gsub("if (data%[%d%])(%[2%] then)", "if %1 and %1%2")

assert(load(code, path))()

local files = {
  "../../PCPT",
  "../../PCPT_01_getstarted",
  "../../PCPT_02_classes",
  "../../PCPT_03_tensor",
  "../../PCPT_04_grad",
  "../../PCPT_05_nn",
  "../../PCPT_06_convolution",
  "../../PCPT_07_classification",
  "../../PCPT_08_training",
  "../../PCPT_09_recursion",
  "../../PCPT_10_essential",
}

for _, f in ipairs(files) do
  local ok, err = pcall(newpax.writenewpax, f)
  if ok then
    print("OK: " .. f .. ".newpax")
  else
    print("FAILED: " .. f .. " -- " .. tostring(err))
  end
end
