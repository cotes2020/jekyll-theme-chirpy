# frozen_string_literal: true

module Sass
  # The {ELF} class.
  #
  # It parses ELF header to extract interpreter.
  # @see https://github.com/torvalds/linux/blob/HEAD/include/uapi/linux/elf.h
  # @see https://github.com/torvalds/linux/blob/HEAD/kernel/kexec_elf.c
  class ELF
    # rubocop:disable Naming/ConstantName

    # 32-bit ELF base types.
    Elf32_Addr  = :__u32
    Elf32_Half  = :__u16
    Elf32_Off   = :__u32
    Elf32_Sword = :__s32
    Elf32_Word  = :__u32

    # 64-bit ELF base types.
    Elf64_Addr   = :__u64
    Elf64_Half   = :__u16
    Elf64_SHalf  = :__s16
    Elf64_Off    = :__u64
    Elf64_Sword  = :__s32
    Elf64_Word   = :__u32
    Elf64_Xword  = :__u64
    Elf64_Sxword = :__s64

    # rubocop:enable Naming/ConstantName

    # These constants are for the segment types stored in the image headers
    PT_NULL         = 0
    PT_LOAD         = 1
    PT_DYNAMIC      = 2
    PT_INTERP       = 3
    PT_NOTE         = 4
    PT_SHLIB        = 5
    PT_PHDR         = 6
    PT_TLS          = 7
    PT_LOOS         = 0x60000000
    PT_HIOS         = 0x6fffffff
    PT_LOPROC       = 0x70000000
    PT_HIPROC       = 0x7fffffff
    PT_GNU_EH_FRAME = (PT_LOOS + 0x474e550)
    PT_GNU_STACK    = (PT_LOOS + 0x474e551)
    PT_GNU_RELRO    = (PT_LOOS + 0x474e552)
    PT_GNU_PROPERTY = (PT_LOOS + 0x474e553)

    # These constants define the different elf file types
    ET_NONE   = 0
    ET_REL    = 1
    ET_EXEC   = 2
    ET_DYN    = 3
    ET_CORE   = 4
    ET_LOPROC = 0xff00
    ET_HIPROC = 0xffff

    EI_NIDENT = 16

    # rubocop:disable Naming/ConstantName

    Elf32_Ehdr = [
      [:unsigned_char, :e_ident, EI_NIDENT],
      [Elf32_Half,     :e_type],
      [Elf32_Half,     :e_machine],
      [Elf32_Word,     :e_version],
      [Elf32_Addr,     :e_entry],
      [Elf32_Off,      :e_phoff],
      [Elf32_Off,      :e_shoff],
      [Elf32_Word,     :e_flags],
      [Elf32_Half,     :e_ehsize],
      [Elf32_Half,     :e_phentsize],
      [Elf32_Half,     :e_phnum],
      [Elf32_Half,     :e_shentsize],
      [Elf32_Half,     :e_shnum],
      [Elf32_Half,     :e_shstrndx]
    ].freeze

    Elf64_Ehdr = [
      [:unsigned_char, :e_ident, EI_NIDENT],
      [Elf64_Half,     :e_type],
      [Elf64_Half,     :e_machine],
      [Elf64_Word,     :e_version],
      [Elf64_Addr,     :e_entry],
      [Elf64_Off,      :e_phoff],
      [Elf64_Off,      :e_shoff],
      [Elf64_Word,     :e_flags],
      [Elf64_Half,     :e_ehsize],
      [Elf64_Half,     :e_phentsize],
      [Elf64_Half,     :e_phnum],
      [Elf64_Half,     :e_shentsize],
      [Elf64_Half,     :e_shnum],
      [Elf64_Half,     :e_shstrndx]
    ].freeze

    Elf32_Phdr = [
      [Elf32_Word, :p_type],
      [Elf32_Off,  :p_offset],
      [Elf32_Addr, :p_vaddr],
      [Elf32_Addr, :p_paddr],
      [Elf32_Word, :p_filesz],
      [Elf32_Word, :p_memsz],
      [Elf32_Word, :p_flags],
      [Elf32_Word, :p_align]
    ].freeze

    Elf64_Phdr = [
      [Elf64_Word,  :p_type],
      [Elf64_Word,  :p_flags],
      [Elf64_Off,   :p_offset],
      [Elf64_Addr,  :p_vaddr],
      [Elf64_Addr,  :p_paddr],
      [Elf64_Xword, :p_filesz],
      [Elf64_Xword, :p_memsz],
      [Elf64_Xword, :p_align]
    ].freeze

    # rubocop:enable Naming/ConstantName

    # e_ident[] indexes
    EI_MAG0    = 0
    EI_MAG1    = 1
    EI_MAG2    = 2
    EI_MAG3    = 3
    EI_CLASS   = 4
    EI_DATA    = 5
    EI_VERSION = 6
    EI_OSABI   = 7
    EI_PAD     = 8

    # EI_MAG
    ELFMAG0 = 0x7f
    ELFMAG1 = 'E'.ord
    ELFMAG2 = 'L'.ord
    ELFMAG3 = 'F'.ord
    ELFMAG  = [ELFMAG0, ELFMAG1, ELFMAG2, ELFMAG3].pack('C*')
    SELFMAG = 4

    # e_ident[EI_CLASS]
    ELFCLASSNONE = 0
    ELFCLASS32   = 1
    ELFCLASS64   = 2
    ELFCLASSNUM  = 3

    # e_ident[EI_DATA]
    ELFDATANONE = 0
    ELFDATA2LSB = 1
    ELFDATA2MSB = 2

    def initialize(buffer)
      @buffer = buffer
      @ehdr = read_ehdr
      @buffer.seek(@ehdr[:e_phoff], IO::SEEK_SET)
      @proghdrs = Array.new(@ehdr[:e_phnum]) do
        read_phdr
      end
    end

    def relocatable?
      @ehdr[:e_type] == ET_REL
    end

    def executable?
      @ehdr[:e_type] == ET_EXEC
    end

    def shared_object?
      @ehdr[:e_type] == ET_DYN
    end

    def core?
      @ehdr[:e_type] == ET_CORE
    end

    def interpreter
      phdr = @proghdrs.find { |p| p[:p_type] == PT_INTERP }
      return if phdr.nil?

      @buffer.seek(phdr[:p_offset], IO::SEEK_SET)
      interpreter = @buffer.read(phdr[:p_filesz])
      raise ArgumentError unless interpreter.end_with?("\0")

      interpreter.chomp!("\0")
    end

    private

    def file_class
      @ehdr[:e_ident][EI_CLASS]
    end

    def data_encoding
      @ehdr[:e_ident][EI_DATA]
    end

    def read_ehdr
      @ehdr = { e_ident: @buffer.read(EI_NIDENT).unpack('C*') }
      raise ArgumentError unless @ehdr[:e_ident].slice(EI_MAG0, SELFMAG).pack('C*') == ELFMAG

      case file_class
      when ELFCLASS32
        Elf32_Ehdr
      when ELFCLASS64
        Elf64_Ehdr
      else
        raise ArgumentError
      end.drop(1).to_h do |field|
        [field[1], read1(field[0])]
      end.merge!(@ehdr)
    end

    def read_phdr
      case file_class
      when ELFCLASS32
        Elf32_Phdr
      when ELFCLASS64
        Elf64_Phdr
      else
        raise ArgumentError
      end.to_h do |field|
        [field[1], read1(field[0])]
      end
    end

    def explicit_endian
      case data_encoding
      when ELFDATA2LSB
        '<'
      when ELFDATA2MSB
        '>'
      else
        raise ArgumentError
      end
    end

    def read1(type)
      case type
      when :__u8
        @buffer.read(1).unpack1('C')
      when :__u16
        @buffer.read(2).unpack1("S#{explicit_endian}")
      when :__u32
        @buffer.read(4).unpack1("L#{explicit_endian}")
      when :__u64
        @buffer.read(8).unpack1("Q#{explicit_endian}")
      when :__s8
        @buffer.read(1).unpack1('c')
      when :__s16
        @buffer.read(2).unpack1("s#{explicit_endian}")
      when :__s32
        @buffer.read(4).unpack1("l#{explicit_endian}")
      when :__s64
        @buffer.read(8).unpack1("q#{explicit_endian}")
      else
        raise ArgumentError
      end
    end

    INTERPRETER = begin
      proc_self_exe = '/proc/self/exe'
      if File.exist?(proc_self_exe)
        File.open(proc_self_exe, 'rb') do |file|
          elf = ELF.new(file)
          interpreter = elf.interpreter
          if interpreter.nil? && elf.shared_object?
            File.readlink(proc_self_exe)
          else
            interpreter
          end
        end
      end
    end
  end

  private_constant :ELF
end
