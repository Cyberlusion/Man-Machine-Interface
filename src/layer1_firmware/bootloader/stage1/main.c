#include <stdint.h>
#include <stdbool.h>

extern void secure_boot(void);

void Reset_Handler(void)
{
    // Initialize RAM
    extern uint32_t _sdata, _edata, _sbss, _ebss, _sidata;
    uint32_t *src = &_sidata;
    uint32_t *dst = &_sdata;

    while (dst < &_edata)
    {
        *dst++ = *src++;
    }

    dst = &_sbss;
    while (dst < &_ebss)
    {
        *dst++ = 0;
    }

    // Jump to secure boot
    secure_boot();

    // Should never reach here
    while (1)
        ;
}