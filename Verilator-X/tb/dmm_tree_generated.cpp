#include "dmm_tree_generated.h"

DataflowTemplate dmm_tree_infer_impl(const DMMFeatures& f) {
    if (f.kernel <= 0.500000f) {
        return DataflowTemplate::ELEMENTWISE;
    } else {
        if (f.op_type <= 0.500000f) {
            if (f.in_ch <= 48.000000f) {
                if (f.buffer_kb <= 48.000000f) {
                    if (f.kernel <= 4.000000f) {
                        if (f.kernel <= 2.000000f) {
                            if (f.out_ch <= 448.000000f) {
                                if (f.out_ch <= 320.000000f) {
                                    return DataflowTemplate::BLOCK;
                                } else {
                                    if (f.tiles <= 1.500000f) {
                                        return DataflowTemplate::BLOCK;
                                    } else {
                                        return DataflowTemplate::BLOCK;
                                    }
                                }
                            } else {
                                if (f.buffer_kb <= 12.000000f) {
                                    return DataflowTemplate::BLOCK;
                                } else {
                                    return DataflowTemplate::BLOCK;
                                }
                            }
                        } else {
                            if (f.out_ch <= 112.000000f) {
                                if (f.in_ch <= 24.000000f) {
                                    if (f.buffer_kb <= 12.000000f) {
                                        if (f.in_ch <= 12.000000f) {
                                            return DataflowTemplate::BLOCK;
                                        } else {
                                            return DataflowTemplate::BLOCK;
                                        }
                                    } else {
                                        return DataflowTemplate::BLOCK;
                                    }
                                } else {
                                    if (f.buffer_kb <= 12.000000f) {
                                        if (f.out_ch <= 40.000000f) {
                                            return DataflowTemplate::BLOCK;
                                        } else {
                                            return DataflowTemplate::LINE;
                                        }
                                    } else {
                                        if (f.pe_cols <= 24.000000f) {
                                            return DataflowTemplate::BLOCK;
                                        } else {
                                            return DataflowTemplate::BLOCK;
                                        }
                                    }
                                }
                            } else {
                                if (f.in_ch <= 5.500000f) {
                                    return DataflowTemplate::BLOCK;
                                } else {
                                    if (f.out_ch <= 224.000000f) {
                                        if (f.in_ch <= 12.000000f) {
                                            return DataflowTemplate::BLOCK;
                                        } else {
                                            return DataflowTemplate::LINE;
                                        }
                                    } else {
                                        if (f.pe_cols <= 192.000000f) {
                                            return DataflowTemplate::LINE;
                                        } else {
                                            return DataflowTemplate::LINE;
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if (f.out_ch <= 40.000000f) {
                            if (f.in_ch <= 12.000000f) {
                                return DataflowTemplate::BLOCK;
                            } else {
                                if (f.kernel <= 6.000000f) {
                                    if (f.buffer_kb <= 12.000000f) {
                                        return DataflowTemplate::BLOCK;
                                    } else {
                                        return DataflowTemplate::BLOCK;
                                    }
                                } else {
                                    if (f.out_ch <= 12.000000f) {
                                        return DataflowTemplate::BLOCK;
                                    } else {
                                        if (f.buffer_kb <= 24.000000f) {
                                            return DataflowTemplate::LINE;
                                        } else {
                                            return DataflowTemplate::BLOCK;
                                        }
                                    }
                                }
                            }
                        } else {
                            if (f.in_ch <= 5.500000f) {
                                if (f.out_ch <= 112.000000f) {
                                    if (f.pe_rows <= 12.000000f) {
                                        return DataflowTemplate::BLOCK;
                                    } else {
                                        return DataflowTemplate::BLOCK;
                                    }
                                } else {
                                    if (f.buffer_kb <= 24.000000f) {
                                        return DataflowTemplate::LINE;
                                    } else {
                                        return DataflowTemplate::LINE;
                                    }
                                }
                            } else {
                                if (f.in_ch <= 12.000000f) {
                                    if (f.out_ch <= 80.000000f) {
                                        if (f.buffer_kb <= 12.000000f) {
                                            return DataflowTemplate::LINE;
                                        } else {
                                            return DataflowTemplate::BLOCK;
                                        }
                                    } else {
                                        if (f.out_ch <= 112.000000f) {
                                            return DataflowTemplate::LINE;
                                        } else {
                                            return DataflowTemplate::LINE;
                                        }
                                    }
                                } else {
                                    return DataflowTemplate::LINE;
                                }
                            }
                        }
                    }
                } else {
                    if (f.out_ch <= 160.000000f) {
                        if (f.kernel <= 6.000000f) {
                            if (f.in_h <= 6.000000f) {
                                if (f.in_ch <= 24.000000f) {
                                    return DataflowTemplate::BLOCK;
                                } else {
                                    if (f.buffer_kb <= 96.000000f) {
                                        return DataflowTemplate::BLOCK;
                                    } else {
                                        return DataflowTemplate::BLOCK;
                                    }
                                }
                            } else {
                                return DataflowTemplate::BLOCK;
                            }
                        } else {
                            if (f.out_ch <= 112.000000f) {
                                if (f.in_ch <= 24.000000f) {
                                    return DataflowTemplate::BLOCK;
                                } else {
                                    if (f.buffer_kb <= 96.000000f) {
                                        return DataflowTemplate::LINE;
                                    } else {
                                        if (f.tiles <= 96.000000f) {
                                            return DataflowTemplate::BLOCK;
                                        } else {
                                            return DataflowTemplate::BLOCK;
                                        }
                                    }
                                }
                            } else {
                                if (f.in_h <= 48.000000f) {
                                    if (f.in_h <= 12.000000f) {
                                        return DataflowTemplate::BLOCK;
                                    } else {
                                        return DataflowTemplate::BLOCK;
                                    }
                                } else {
                                    return DataflowTemplate::LINE;
                                }
                            }
                        }
                    } else {
                        if (f.kernel <= 6.000000f) {
                            if (f.in_ch <= 12.000000f) {
                                return DataflowTemplate::BLOCK;
                            } else {
                                if (f.kernel <= 2.000000f) {
                                    return DataflowTemplate::BLOCK;
                                } else {
                                    if (f.buffer_kb <= 192.000000f) {
                                        if (f.out_ch <= 224.000000f) {
                                            return DataflowTemplate::BLOCK;
                                        } else {
                                            return DataflowTemplate::LINE;
                                        }
                                    } else {
                                        if (f.in_h <= 48.000000f) {
                                            return DataflowTemplate::BLOCK;
                                        } else {
                                            return DataflowTemplate::BLOCK;
                                        }
                                    }
                                }
                            }
                        } else {
                            if (f.in_ch <= 5.500000f) {
                                return DataflowTemplate::BLOCK;
                            } else {
                                if (f.buffer_kb <= 192.000000f) {
                                    if (f.in_ch <= 12.000000f) {
                                        if (f.buffer_kb <= 96.000000f) {
                                            return DataflowTemplate::LINE;
                                        } else {
                                            return DataflowTemplate::BLOCK;
                                        }
                                    } else {
                                        return DataflowTemplate::LINE;
                                    }
                                } else {
                                    if (f.in_ch <= 24.000000f) {
                                        if (f.buffer_kb <= 384.000000f) {
                                            return DataflowTemplate::BLOCK;
                                        } else {
                                            return DataflowTemplate::BLOCK;
                                        }
                                    } else {
                                        if (f.in_w <= 48.000000f) {
                                            return DataflowTemplate::LINE;
                                        } else {
                                            return DataflowTemplate::LINE;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                if (f.kernel <= 2.000000f) {
                    if (f.buffer_kb <= 24.000000f) {
                        if (f.out_ch <= 112.000000f) {
                            if (f.in_ch <= 384.000000f) {
                                if (f.out_ch <= 56.000000f) {
                                    return DataflowTemplate::BLOCK;
                                } else {
                                    if (f.in_ch <= 192.000000f) {
                                        return DataflowTemplate::BLOCK;
                                    } else {
                                        return DataflowTemplate::LINE;
                                    }
                                }
                            } else {
                                if (f.out_ch <= 24.000000f) {
                                    return DataflowTemplate::BLOCK;
                                } else {
                                    if (f.out_ch <= 40.000000f) {
                                        return DataflowTemplate::LINE;
                                    } else {
                                        return DataflowTemplate::LINE;
                                    }
                                }
                            }
                        } else {
                            if (f.out_ch <= 160.000000f) {
                                if (f.buffer_kb <= 12.000000f) {
                                    return DataflowTemplate::LINE;
                                } else {
                                    return DataflowTemplate::BLOCK;
                                }
                            } else {
                                if (f.tiles <= 384.000000f) {
                                    return DataflowTemplate::LINE;
                                } else {
                                    return DataflowTemplate::LINE;
                                }
                            }
                        }
                    } else {
                        if (f.out_ch <= 224.000000f) {
                            if (f.in_ch <= 384.000000f) {
                                return DataflowTemplate::BLOCK;
                            } else {
                                if (f.out_ch <= 112.000000f) {
                                    return DataflowTemplate::BLOCK;
                                } else {
                                    if (f.buffer_kb <= 96.000000f) {
                                        return DataflowTemplate::LINE;
                                    } else {
                                        return DataflowTemplate::BLOCK;
                                    }
                                }
                            }
                        } else {
                            if (f.buffer_kb <= 96.000000f) {
                                if (f.in_ch <= 192.000000f) {
                                    if (f.pe_cols <= 24.000000f) {
                                        return DataflowTemplate::BLOCK;
                                    } else {
                                        return DataflowTemplate::BLOCK;
                                    }
                                } else {
                                    if (f.in_h <= 48.000000f) {
                                        return DataflowTemplate::LINE;
                                    } else {
                                        return DataflowTemplate::LINE;
                                    }
                                }
                            } else {
                                if (f.in_ch <= 384.000000f) {
                                    return DataflowTemplate::BLOCK;
                                } else {
                                    return DataflowTemplate::BLOCK;
                                }
                            }
                        }
                    }
                } else {
                    if (f.buffer_kb <= 96.000000f) {
                        if (f.out_ch <= 24.000000f) {
                            if (f.in_ch <= 192.000000f) {
                                if (f.buffer_kb <= 24.000000f) {
                                    if (f.in_ch <= 96.000000f) {
                                        if (f.kernel <= 4.000000f) {
                                            return DataflowTemplate::BLOCK;
                                        } else {
                                            return DataflowTemplate::LINE;
                                        }
                                    } else {
                                        return DataflowTemplate::LINE;
                                    }
                                } else {
                                    if (f.kernel <= 6.000000f) {
                                        if (f.in_h <= 12.000000f) {
                                            return DataflowTemplate::BLOCK;
                                        } else {
                                            return DataflowTemplate::BLOCK;
                                        }
                                    } else {
                                        return DataflowTemplate::LINE;
                                    }
                                }
                            } else {
                                if (f.in_ch <= 384.000000f) {
                                    if (f.buffer_kb <= 24.000000f) {
                                        return DataflowTemplate::LINE;
                                    } else {
                                        return DataflowTemplate::LINE;
                                    }
                                } else {
                                    return DataflowTemplate::LINE;
                                }
                            }
                        } else {
                            if (f.in_ch <= 96.000000f) {
                                if (f.out_ch <= 56.000000f) {
                                    if (f.kernel <= 4.000000f) {
                                        if (f.buffer_kb <= 24.000000f) {
                                            return DataflowTemplate::LINE;
                                        } else {
                                            return DataflowTemplate::BLOCK;
                                        }
                                    } else {
                                        if (f.tiles <= 384.000000f) {
                                            return DataflowTemplate::LINE;
                                        } else {
                                            return DataflowTemplate::LINE;
                                        }
                                    }
                                } else {
                                    if (f.buffer_kb <= 48.000000f) {
                                        return DataflowTemplate::LINE;
                                    } else {
                                        if (f.kernel <= 4.000000f) {
                                            return DataflowTemplate::LINE;
                                        } else {
                                            return DataflowTemplate::LINE;
                                        }
                                    }
                                }
                            } else {
                                return DataflowTemplate::LINE;
                            }
                        }
                    } else {
                        if (f.out_ch <= 56.000000f) {
                            if (f.kernel <= 6.000000f) {
                                if (f.in_ch <= 384.000000f) {
                                    if (f.buffer_kb <= 192.000000f) {
                                        if (f.out_ch <= 24.000000f) {
                                            return DataflowTemplate::BLOCK;
                                        } else {
                                            return DataflowTemplate::BLOCK;
                                        }
                                    } else {
                                        return DataflowTemplate::BLOCK;
                                    }
                                } else {
                                    if (f.buffer_kb <= 192.000000f) {
                                        return DataflowTemplate::LINE;
                                    } else {
                                        if (f.kernel <= 4.000000f) {
                                            return DataflowTemplate::BLOCK;
                                        } else {
                                            return DataflowTemplate::BLOCK;
                                        }
                                    }
                                }
                            } else {
                                if (f.out_ch <= 24.000000f) {
                                    if (f.tiles <= 48.000000f) {
                                        return DataflowTemplate::LINE;
                                    } else {
                                        return DataflowTemplate::BLOCK;
                                    }
                                } else {
                                    if (f.in_ch <= 96.000000f) {
                                        if (f.out_ch <= 40.000000f) {
                                            return DataflowTemplate::BLOCK;
                                        } else {
                                            return DataflowTemplate::BLOCK;
                                        }
                                    } else {
                                        if (f.in_ch <= 192.000000f) {
                                            return DataflowTemplate::LINE;
                                        } else {
                                            return DataflowTemplate::LINE;
                                        }
                                    }
                                }
                            }
                        } else {
                            if (f.in_ch <= 192.000000f) {
                                if (f.kernel <= 6.000000f) {
                                    if (f.buffer_kb <= 384.000000f) {
                                        if (f.out_ch <= 160.000000f) {
                                            return DataflowTemplate::BLOCK;
                                        } else {
                                            return DataflowTemplate::LINE;
                                        }
                                    } else {
                                        if (f.out_ch <= 160.000000f) {
                                            return DataflowTemplate::BLOCK;
                                        } else {
                                            return DataflowTemplate::BLOCK;
                                        }
                                    }
                                } else {
                                    if (f.out_ch <= 80.000000f) {
                                        return DataflowTemplate::BLOCK;
                                    } else {
                                        if (f.pe_cols <= 24.000000f) {
                                            return DataflowTemplate::LINE;
                                        } else {
                                            return DataflowTemplate::LINE;
                                        }
                                    }
                                }
                            } else {
                                if (f.kernel <= 4.000000f) {
                                    if (f.buffer_kb <= 384.000000f) {
                                        if (f.out_ch <= 80.000000f) {
                                            return DataflowTemplate::LINE;
                                        } else {
                                            return DataflowTemplate::LINE;
                                        }
                                    } else {
                                        if (f.tiles <= 12.000000f) {
                                            return DataflowTemplate::LINE;
                                        } else {
                                            return DataflowTemplate::BLOCK;
                                        }
                                    }
                                } else {
                                    return DataflowTemplate::LINE;
                                }
                            }
                        }
                    }
                }
            }
        } else {
            if (f.kernel <= 2.500000f) {
                return DataflowTemplate::POOL_STREAM;
            } else {
                if (f.out_ch <= 384.000000f) {
                    return DataflowTemplate::BLOCK;
                } else {
                    if (f.buffer_kb <= 12.000000f) {
                        if (f.kernel <= 4.000000f) {
                            return DataflowTemplate::BLOCK;
                        } else {
                            return DataflowTemplate::LINE;
                        }
                    } else {
                        return DataflowTemplate::BLOCK;
                    }
                }
            }
        }
    }
}
