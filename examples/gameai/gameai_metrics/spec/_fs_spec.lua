-- gameai_metrics/spec/_fs_spec.lua
--
-- Unit spec for the tiny POSIX filesystem helper module. Two surfaces:
--
--   * shell_squote(s)    — pure string transform, exhaustively covered
--                          for the shell metacharacters `sh` would
--                          expand outside of single quotes.
--   * ensure_parent_dir  — actually shells out via os.execute; we
--                          exercise the happy path against real temp
--                          directories and stub os.execute for the
--                          loud-error path so the test does not need a
--                          filesystem that can refuse mkdir.

local describe, it, expect = lust.describe, lust.it, lust.expect

local fs = require("gameai_metrics._fs")

-- Fresh path under the tmp root. os.tmpname reserves a file we
-- immediately unlink; we only need the unique prefix.
local _tmp_counter = 0
local function tmp_prefix(tag)
    _tmp_counter = _tmp_counter + 1
    local base = os.tmpname()
    os.remove(base)
    return base .. "-fs-" .. tag .. "-" .. tostring(_tmp_counter)
end

local function dir_exists(path)
    -- POSIX portable "does this directory exist" test without lfs:
    -- opening the trailing "/." succeeds only for directories.
    local f = io.open(path .. "/.", "r")
    if f == nil then
        return false
    end
    f:close()
    return true
end

describe("gameai_metrics._fs.shell_squote", function()
    it("wraps an empty string in a pair of single quotes", function()
        expect(fs.shell_squote("")).to.equal("''")
    end)

    it("wraps a plain path unchanged", function()
        expect(fs.shell_squote("/tmp/harvest.json")).to.equal("'/tmp/harvest.json'")
    end)

    it("preserves spaces inside the quoted form", function()
        expect(fs.shell_squote("/tmp/has space/x")).to.equal("'/tmp/has space/x'")
    end)

    it("neutralises $ / backtick / ! metachars by keeping them inside single quotes", function()
        expect(fs.shell_squote("/tmp/$HOME")).to.equal("'/tmp/$HOME'")
        expect(fs.shell_squote("/tmp/`whoami`")).to.equal("'/tmp/`whoami`'")
        expect(fs.shell_squote("/tmp/foo!")).to.equal("'/tmp/foo!'")
    end)

    it("does not escape a double quote inside single quotes", function()
        -- Inside '...' the shell does not expand ", so the quoted form
        -- keeps the " character verbatim.
        expect(fs.shell_squote('/tmp/say "hi"')).to.equal([['/tmp/say "hi"']])
    end)

    it("rewrites an embedded single quote as '\\''", function()
        expect(fs.shell_squote("it's")).to.equal([['it'\''s']])
    end)

    it("rewrites multiple embedded single quotes", function()
        expect(fs.shell_squote("a'b'c")).to.equal([['a'\''b'\''c']])
    end)

    it("keeps a newline inside the single-quoted form", function()
        expect(fs.shell_squote("line1\nline2")).to.equal("'line1\nline2'")
    end)

    it("refuses a non-string argument", function()
        local ok, err = pcall(fs.shell_squote, 42)
        expect(ok).to.equal(false)
        expect(err:find("string") ~= nil).to.equal(true)
    end)
end)

describe("gameai_metrics._fs.ensure_parent_dir", function()
    it("creates a missing parent directory", function()
        local root = tmp_prefix("mkdir")
        local path = root .. "/nested/deep/file.json"
        expect(dir_exists(root)).to.equal(false)
        fs.ensure_parent_dir(path)
        expect(dir_exists(root .. "/nested/deep")).to.equal(true)
        -- clean up (deepest first)
        os.remove(root .. "/nested/deep")
        os.remove(root .. "/nested")
        os.remove(root)
    end)

    it("is a no-op when the parent already exists", function()
        local root = tmp_prefix("exists")
        assert(os.execute("mkdir -p '" .. root .. "'"))
        expect(dir_exists(root)).to.equal(true)
        -- Should not raise, should not disturb the pre-existing dir.
        fs.ensure_parent_dir(root .. "/file.json")
        expect(dir_exists(root)).to.equal(true)
        os.remove(root)
    end)

    it("does nothing when the path is a top-level relative name", function()
        -- `"foo.json"` has no directory component, so no shell-out.
        -- We prove that by stubbing os.execute — the stub would flag a
        -- surprise invocation by setting `called` to true.
        local called = false
        local real_execute = os.execute
        os.execute = function(_) ---@diagnostic disable-line
            called = true
            return true, "exit", 0
        end
        fs.ensure_parent_dir("foo.json")
        os.execute = real_execute
        expect(called).to.equal(false)
    end)

    it("handles a path with shell metachars via the single-quote wrap", function()
        -- The directory name embeds a space, `$`, and backtick; if the
        -- helper failed to quote we'd either get "mkdir: extra operand"
        -- or a subshell expansion producing the wrong directory.
        local root = tmp_prefix("meta") .. " dir$one`two"
        local path = root .. "/file.json"
        fs.ensure_parent_dir(path)
        expect(dir_exists(root)).to.equal(true)
        os.remove(root)
    end)

    it("raises loudly when os.execute reports a non-ok result", function()
        local real_execute = os.execute
        os.execute = function(_) ---@diagnostic disable-line
            return nil, "exit", 1
        end
        local ok, err = pcall(fs.ensure_parent_dir, "/some/where/file.json")
        os.execute = real_execute
        expect(ok).to.equal(false)
        expect(err:find("mkdir %-p") ~= nil).to.equal(true)
        expect(err:find("failed") ~= nil).to.equal(true)
    end)

    it("refuses a non-string path", function()
        local ok, err = pcall(fs.ensure_parent_dir, 42)
        expect(ok).to.equal(false)
        expect(err:find("string") ~= nil).to.equal(true)
    end)
end)
