<?php
namespace Rindow\NeuralNetworks\Layer;

class GlobalAveragePooling2D extends AbstractGlobalAveragePooling
{
    protected $rank = 2;

    public function __construct(object $backend,array $options=null)
    {
        $leftargs = [];
        extract($this->extractArgs([
            'name'=>null,
        ],$options,$leftargs));
        $this->initName($name,'globalaveragepooling2d');
        parent::__construct($backend, $leftargs);
    }
}
