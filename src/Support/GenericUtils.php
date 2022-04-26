<?php
namespace Rindow\NeuralNetworks\Support;

use InvalidArgumentException;

trait GenericUtils
{
    public static $nameNumbering = 0;

    protected function extractArgs(
        array $keywords,
        array $kwargs=null,
        array &$leftargs=null)
    {
        if($kwargs===null) {
            $kwargs = [];
        }
        $vars = [];
        foreach ($kwargs as $key => $value) {
            if(array_key_exists($key, $keywords)) {
                $vars[$key] = $value;
            } else {
                if($leftargs===null) {
                    throw new InvalidArgumentException(
                        'Unexpected keyword argument "'.$key.'"');
                } else {
                    $leftargs[$key] = $value;
                }
            }
        }
        foreach ($keywords as $key => $default) {
            if(!array_key_exists($key,$kwargs)) {
                $vars[$key] = $default;
            }
        }
        return $vars;
    }

    protected function initName($name,$default)
    {
        if($name===null) {
            if(self::$nameNumbering==0) {
                $name = $default;
            } else {
                $name = $default.'_'.self::$nameNumbering;
            }
            self::$nameNumbering++;
        }
        $this->name = $name;
    }
}
