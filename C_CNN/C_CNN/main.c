//
//  main.c
//  C_CNN
//
//  Created by SHU WANG on 8/26/19.
//  Copyright © 2019 SHU WANG. All rights reserved.
//

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdnoreturn.h>
#include <stdlib.h>
#include <assert.h>
#include <complex.h>

#ifndef var
#define var     __auto_type
#endif

/// noreturn测试函数
static int noreturn MyTest(void)
{
    const bool equal = sizeof(long) == sizeof(int);
    printf("Is equal? %s\n", equal ? "Yes" : "No");
    
    // 原子操作测试
    volatile atomic_int atom;
    atomic_init(&atom, 0);
    const var oldValue = atomic_fetch_add(&atom, 100);
    printf("Previous value = %d, current value = %d\n", oldValue, atomic_load(&atom));
    
    volatile atomic_flag flag = ATOMIC_FLAG_INIT;
    if(!atomic_flag_test_and_set(&flag))
        puts("Access the resource...");
    else
        puts("Locked!");
    
    if(!atomic_flag_test_and_set(&flag))
        puts("Access the resource...");
    else
        puts("Locked!");
    
    atomic_flag_clear(&flag);
    if(!atomic_flag_test_and_set(&flag))
        puts("Access the resource...");
    else
        puts("Locked!");
    
    exit(0);
}

int main(int argc, const char * argv[])
{
    // 字符编码测试
    const char *utf8Str = u8"你好，世界！";
    printf("The string is: %s\n", utf8Str);
    
    const var utf16Char = u'¥';
    printf("The UTF-16 code is: %.4X\n", utf16Char);
    
    const char encoding = _Generic(utf16Char, uint8_t:'c',
                                   uint16_t:'s', uint32_t:'i', default:'o');
    
    printf("The type of utf16Char is: %c\n", encoding);
    
    // 复数测试
    complex float comp1 = 5.0f + 3.0fi;
    complex float comp2 = {3.0f, -2.0f};
    comp1 -= comp2;
    printf("comp1 real = %f, imag = %f\n", __real(comp1), __imag(comp1));
    
    // 静态断言测试
    static_assert(sizeof(long) == 8, "Not 64-bit environment!");
    
    return MyTest();
}
