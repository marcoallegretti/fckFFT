#ifndef FCKFFT_TESTS_NPZ_H
#define FCKFFT_TESTS_NPZ_H

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace npz {

struct Array
{
    std::vector<std::size_t> shape;
    std::vector<float> data;
};

inline std::uint32_t le32(const unsigned char* p)
{
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8)
           | (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

inline std::uint16_t le16(const unsigned char* p)
{
    return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}

inline bool parseNpy(const unsigned char* p, std::size_t size, Array& out)
{
    if (size < 10 || std::memcmp(p, "\x93NUMPY", 6) != 0)
        return false;
    std::size_t headerLen, offset;
    if (p[6] == 1)
    {
        headerLen = le16(p + 8);
        offset = 10;
    }
    else
    {
        headerLen = le32(p + 8);
        offset = 12;
    }
    const std::string header(reinterpret_cast<const char*>(p + offset), headerLen);
    if (header.find("'<f4'") == std::string::npos || header.find("'fortran_order': False") == std::string::npos)
        return false;
    const std::size_t open = header.find('(', header.find("'shape'"));
    const std::size_t close = header.find(')', open);
    out.shape.clear();
    std::size_t count = 1;
    std::size_t i = open + 1;
    while (i < close)
    {
        while (i < close && (header[i] < '0' || header[i] > '9'))
            ++i;
        if (i >= close)
            break;
        std::size_t v = 0;
        while (i < close && header[i] >= '0' && header[i] <= '9')
            v = v * 10 + static_cast<std::size_t>(header[i++] - '0');
        out.shape.push_back(v);
        count *= v;
    }
    const std::size_t dataOffset = offset + headerLen;
    if (dataOffset + count * 4 > size)
        return false;
    out.data.resize(count);
    for (std::size_t k = 0; k < count; ++k)
    {
        const std::uint32_t bits = le32(p + dataOffset + 4 * k);
        std::memcpy(&out.data[k], &bits, 4);
    }
    return true;
}

inline bool load(const std::string& path, const std::string& name, Array& out)
{
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr)
        return false;
    std::vector<unsigned char> bytes;
    unsigned char buffer[4096];
    std::size_t n;
    while ((n = std::fread(buffer, 1, sizeof buffer, f)) > 0)
        bytes.insert(bytes.end(), buffer, buffer + n);
    std::fclose(f);
    std::size_t pos = 0;
    const std::string wanted = name + ".npy";
    while (pos + 30 <= bytes.size() && le32(&bytes[pos]) == 0x04034b50u)
    {
        const std::uint16_t method = le16(&bytes[pos + 8]);
        std::size_t compressed = le32(&bytes[pos + 18]);
        const std::uint16_t nameLen = le16(&bytes[pos + 26]);
        const std::uint16_t extraLen = le16(&bytes[pos + 28]);
        const std::string entry(reinterpret_cast<const char*>(&bytes[pos + 30]), nameLen);
        const std::size_t extraStart = pos + 30 + nameLen;
        const std::size_t dataStart = extraStart + extraLen;
        if (dataStart > bytes.size())
            return false;
        for (std::size_t e = extraStart; e + 4 <= dataStart;)
        {
            const std::uint16_t id = le16(&bytes[e]);
            const std::uint16_t len = le16(&bytes[e + 2]);
            if (id == 0x0001 && len >= 16 && compressed == 0xFFFFFFFFu)
                compressed = static_cast<std::size_t>(le32(&bytes[e + 12]))
                             | (static_cast<std::size_t>(le32(&bytes[e + 16])) << 16 << 16);
            e += 4 + static_cast<std::size_t>(len);
        }
        if (dataStart + compressed > bytes.size())
            return false;
        if (entry == wanted)
            return method == 0 && parseNpy(&bytes[dataStart], compressed, out);
        pos = dataStart + compressed;
    }
    return false;
}

}

#endif
